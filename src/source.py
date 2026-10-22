"""Источник задач, пакетно читает большой файл и публикует задачи в очередь

Источник публикует быстрее, чем воркер обрабатывает
Когда очередь заполнена, брокер отказывает в публикации, источник ждёт и повторяет ту же задачу,
поэтому его скорость сама подстраивается под скорость воркеров
Задачи не теряются и не копятся в памяти, в каждый момент источник держит одну задачу
и читает следующую строку файла только после подтверждения брокера
"""

import argparse
import asyncio
import contextlib
import csv
import logging
import signal
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from aiormq.exceptions import (
    AMQPConnectionError,
    ChannelInvalidStateError,
    DeliveryError,
    PublishError,
)
from faststream.rabbit import RabbitBroker
from pydantic import ValidationError

from src.broker import TASKS_EXCHANGE, TASKS_ROUTING_KEY, declare_topology, make_broker
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.schemas import TaskMessage

# Имя задано явно, при запуске через python -m модуль называется __main__
log = logging.getLogger("src.source")

# Пространство имён id задач, одна и та же строка файла в одном запуске всегда даёт один id
TASK_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/Gegerout/ai-audio-censor/tasks")
# Сбой связи с брокером временный, публикацию той же задачи повторяют после паузы
BROKER_ERRORS = (AMQPConnectionError, ChannelInvalidStateError, ConnectionError, TimeoutError)


@dataclass
class SourceTask:
    """Строка файла, которую нужно отправить на обработку"""

    task_id: UUID
    text: str
    row: int


def read_tasks(path: Path, column: str, run: str) -> Iterator[SourceTask]:
    """Читать файл построчно и выдавать задачи по одной, файл целиком в память не попадает"""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"В файле {path.name} нет колонки {column}")
        for row, values in enumerate(reader):
            yield SourceTask(uuid5(TASK_NAMESPACE, f"{run}/{path.name}/{row}"), values[column], row)


@dataclass
class Backoff:
    """Пауза перед публикацией, при отказе брокера растёт вдвое, при успехе вдвое сокращается"""

    initial: float
    maximum: float
    delay: float = 0.0

    def on_reject(self) -> None:
        """Брокер отказал, источник публикует быстрее, чем воркеры освобождают очередь"""
        self.delay = min(self.maximum, max(self.initial, self.delay * 2))

    def on_accept(self) -> None:
        """Брокер принял задачу, место в очереди есть, источник постепенно ускоряется"""
        self.delay = self.delay / 2 if self.delay / 2 >= self.initial else 0.0


@dataclass
class SourceStats:
    """Счётчики источника за весь запуск"""

    published: int = 0
    rejected: int = 0
    unavailable: int = 0
    skipped: int = 0
    # Первая строка, которая ещё не принята брокером, с неё можно продолжить после остановки
    next_row: int = 0


class Source:
    """Публикация задач с подтверждением брокера и повтором после отказа"""

    def __init__(
        self,
        broker: RabbitBroker,
        settings: Settings,
        stop: asyncio.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.broker = broker
        self.backoff = Backoff(
            settings.source_initial_delay_seconds, settings.source_max_delay_seconds
        )
        self.report_seconds = settings.source_report_seconds
        self.stop = stop or asyncio.Event()
        self.clock = clock
        self.stats = SourceStats()
        self._started_at = self._reported_at = clock()
        self._reported_published = self._reported_rejected = 0

    async def run(self, tasks: Iterable[SourceTask]) -> SourceStats:
        """Опубликовать задачи по очереди, следующая строка читается после подтверждения"""
        for task in tasks:
            self.stats.next_row = task.row
            if self.stop.is_set() or not await self.publish(task):
                break
            self.stats.next_row = task.row + 1
        log.info(
            "source stopped" if self.stop.is_set() else "source finished", extra=self._summary()
        )
        return self.stats

    async def publish(self, task: SourceTask) -> bool:
        """Публиковать задачу, пока брокер её не подтвердит, False если источник остановлен"""
        try:
            message = TaskMessage(
                task_id=task.task_id, text=task.text, enqueued_at=datetime.now(UTC)
            )
        except ValidationError as exc:
            self.stats.skipped += 1
            log.warning(
                "invalid row skipped", extra={"row": task.row, "error": exc.errors()[0]["msg"]}
            )
            return True
        while True:
            if self.backoff.delay:
                await self._pause()
            if self.stop.is_set():
                return False
            # Время приёма ставится при каждой попытке, в очередь задача попадает только сейчас
            message = message.model_copy(update={"enqueued_at": datetime.now(UTC)})
            try:
                await self.broker.publish(
                    message,
                    exchange=TASKS_EXCHANGE,
                    routing_key=TASKS_ROUTING_KEY,
                    # Сообщение пишется на диск брокера и переживает его перезапуск
                    persist=True,
                )
            except PublishError:
                # Сообщение не попало ни в одну очередь, ждём, пока очередь задач объявят
                self.stats.unavailable += 1
                self.backoff.on_reject()
                log.warning("tasks queue is missing, retrying", extra=self._progress())
            except DeliveryError:
                # Очередь заполнена, брокер отказал по политике лимита очереди
                self.stats.rejected += 1
                self.backoff.on_reject()
            except BROKER_ERRORS as exc:
                self.stats.unavailable += 1
                self.backoff.on_reject()
                log.warning("broker unavailable, retrying", extra={"error": repr(exc)})
            else:
                self.stats.published += 1
                self.backoff.on_accept()
                self._maybe_report()
                return True
            self._maybe_report()

    async def _pause(self) -> None:
        """Подождать паузу, сигнал остановки прерывает её сразу"""
        # Пауза всегда больше нуля, поэтому повторы не превращаются в холостой цикл
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.stop.wait(), self.backoff.delay)

    def _maybe_report(self) -> None:
        """Раз в report_seconds записать в лог скорость публикации и число отказов"""
        now = self.clock()
        elapsed = now - self._reported_at
        if elapsed < self.report_seconds:
            return
        log.info(
            "source progress",
            extra={
                **self._progress(),
                "rate": round((self.stats.published - self._reported_published) / elapsed, 1),
                "rejected_recently": self.stats.rejected - self._reported_rejected,
            },
        )
        self._reported_at = now
        self._reported_published = self.stats.published
        self._reported_rejected = self.stats.rejected

    def _progress(self) -> dict:
        """Текущие счётчики и пауза источника"""
        return {
            "published": self.stats.published,
            "rejected": self.stats.rejected,
            "delay_ms": round(self.backoff.delay * 1000),
        }

    def _summary(self) -> dict:
        """Итог запуска со средней скоростью публикации и строкой для продолжения"""
        elapsed = max(self.clock() - self._started_at, 1e-9)
        return {
            **self._progress(),
            "unavailable": self.stats.unavailable,
            "skipped": self.stats.skipped,
            "next_row": self.stats.next_row,
            "seconds": round(elapsed, 1),
            "rate": round(self.stats.published / elapsed, 1),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки"""
    parser = argparse.ArgumentParser(description="Пакетная отправка строк файла на обработку")
    parser.add_argument("--file", type=Path, required=True, help="CSV файл с текстами")
    parser.add_argument("--column", default="text_of_post", help="колонка с текстом")
    parser.add_argument("--run", default="default", help="имя запуска, от него зависят id задач")
    parser.add_argument("--start", type=int, default=0, help="с какой строки начать")
    parser.add_argument("--count", type=int, default=None, help="сколько строк отправить")
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> SourceStats:
    """Подключиться к брокеру и отправить строки файла"""
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # По SIGTERM и Ctrl+C источник прерывает паузу, пишет итог и строку для продолжения
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    broker = make_broker(settings)
    await broker.connect()
    try:
        await declare_topology(broker)
        end = None if args.count is None else args.start + args.count
        tasks = islice(read_tasks(args.file, args.column, args.run), args.start, end)
        log.info("source started", extra={"file": str(args.file), "run": args.run})
        return await Source(broker, settings, stop).run(tasks)
    finally:
        await broker.stop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
