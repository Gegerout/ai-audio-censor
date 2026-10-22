"""Тесты источника задач, чтения файла, паузы после отказа и остановки"""

import asyncio
import csv
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid5

import pytest
from aiormq.exceptions import (
    AMQPConnectionError,
    ChannelInvalidStateError,
    DeliveryError,
    PublishError,
)
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from pamqp.commands import Basic

from src import source
from src.broker import TASKS_EXCHANGE, TASKS_QUEUE
from src.config import get_settings
from src.schemas import TaskMessage
from src.source import TASK_NAMESPACE, Backoff, Source, SourceTask, main, read_tasks

TEXTS = ["You are an idiot.", "Have a nice day", "What a stupid idea", "Thanks a lot"]


def queue_full() -> DeliveryError:
    """Отказ брокера, который приходит при заполненной очереди"""
    return DeliveryError(None, Basic.Nack(delivery_tag=1))


def no_route() -> PublishError:
    """Возврат сообщения, которое не попало ни в одну очередь"""
    returned = Mock(delivery=Basic.Return(312, "NO_ROUTE", "tasks", "tasks"))
    return PublishError(returned, None)


@pytest.fixture
def csv_file(tmp_path) -> Path:
    """Файл в формате датасета с текстом в колонке text_of_post"""
    path = tmp_path / "train.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["position", "text_of_post"])
        writer.writeheader()
        writer.writerows({"position": "[]", "text_of_post": text} for text in TEXTS)
    return path


@pytest.fixture
def settings():
    """Короткие паузы источника, чтобы тесты шли быстро"""
    return get_settings().model_copy(
        update={
            "source_initial_delay_seconds": 0.001,
            "source_max_delay_seconds": 0.004,
            "source_report_seconds": 60,
        }
    )


def tasks(*texts: str) -> list[SourceTask]:
    """Задачи с id по номеру строки"""
    return [
        SourceTask(uuid5(TASK_NAMESPACE, f"test/{row}"), text, row)
        for row, text in enumerate(texts)
    ]


def fake_broker(*outcomes: Exception | None) -> Mock:
    """Брокер, который отвечает на публикации по очереди, None значит подтверждение"""
    return Mock(publish=AsyncMock(side_effect=list(outcomes)))


def test_read_tasks_gives_same_ids_for_same_run(csv_file) -> None:
    """Повторный запуск с тем же именем даёт те же id, поэтому повтор не создаёт новых задач"""
    first = list(read_tasks(csv_file, "text_of_post", "exp"))
    again = list(read_tasks(csv_file, "text_of_post", "exp"))
    other = list(read_tasks(csv_file, "text_of_post", "other"))

    assert [task.text for task in first] == TEXTS
    assert [task.row for task in first] == [0, 1, 2, 3]
    assert [task.task_id for task in first] == [task.task_id for task in again]
    assert len({task.task_id for task in first}) == 4
    assert not {task.task_id for task in first} & {task.task_id for task in other}


def test_read_tasks_reads_file_lazily(csv_file) -> None:
    """Строки читаются по одной по мере запроса, а не все сразу"""
    reader = read_tasks(csv_file, "text_of_post", "exp")

    assert next(reader).text == TEXTS[0]
    csv_file.write_text("position,text_of_post\n")
    assert next(reader).text == TEXTS[1]


def test_read_tasks_requires_text_column(csv_file) -> None:
    """Без нужной колонки источник сразу сообщает об ошибке"""
    with pytest.raises(ValueError, match="нет колонки text"):
        next(read_tasks(csv_file, "text", "exp"))


def test_backoff_doubles_up_to_cap_and_halves_on_accept() -> None:
    """Пауза удваивается при отказах до предела и вдвое сокращается при успехах до нуля"""
    backoff = Backoff(initial=0.05, maximum=0.4)
    grown = []
    for _ in range(5):
        backoff.on_reject()
        grown.append(backoff.delay)
    shrunk = []
    for _ in range(4):
        backoff.on_accept()
        shrunk.append(backoff.delay)

    assert grown == [0.05, 0.1, 0.2, 0.4, 0.4]
    assert shrunk == [0.2, 0.1, 0.05, 0.0]


async def test_rejected_task_is_retried_after_growing_pause(settings, monkeypatch) -> None:
    """После отказа брокера источник ждёт и повторяет ту же задачу, пауза растёт до предела"""
    broker = fake_broker(queue_full(), queue_full(), queue_full(), queue_full(), None)
    pauses = []
    worker = Source(broker, settings)
    original = worker._pause

    async def pause():
        pauses.append(worker.backoff.delay)
        await original()

    monkeypatch.setattr(worker, "_pause", pause)
    stats = await worker.run(tasks(TEXTS[0]))

    assert pauses == [0.001, 0.002, 0.004, 0.004]
    assert (stats.published, stats.rejected) == (1, 4)
    sent = [call.args[0] for call in broker.publish.await_args_list]
    assert {message.task_id for message in sent} == {tasks(TEXTS[0])[0].task_id}
    assert all(call.kwargs["persist"] for call in broker.publish.await_args_list)


async def test_source_does_not_read_ahead_while_rejected(settings) -> None:
    """Пока брокер отказывает, следующая строка не читается и не копится в памяти"""
    pulled = []

    def rows():
        for task in tasks(*TEXTS[:2]):
            pulled.append(task.row)
            yield task

    seen = []

    async def publish(message, **kwargs):
        seen.append(list(pulled))
        if len(seen) <= 3:
            raise queue_full()

    broker = Mock(publish=publish)
    stats = await Source(broker, settings).run(rows())

    assert seen[:4] == [[0], [0], [0], [0]]
    assert seen[4] == [0, 1]
    assert stats.published == 2


async def test_accepted_tasks_speed_source_up(settings) -> None:
    """После отказов успешные публикации сокращают паузу, источник снова ускоряется"""
    broker = fake_broker(queue_full(), queue_full(), queue_full(), None, None, None, None)
    worker = Source(broker, settings)

    await worker.run(tasks(*TEXTS))

    assert worker.backoff.delay == 0.0


@pytest.mark.parametrize(
    "error",
    [no_route(), AMQPConnectionError("closed"), ChannelInvalidStateError("closed")],
)
async def test_broker_problems_are_retried(settings, caplog, error) -> None:
    """Нет очереди или связи с брокером, источник ждёт и повторяет, задача не теряется"""
    broker = fake_broker(error, None)

    stats = await Source(broker, settings).run(tasks(TEXTS[0]))

    assert (stats.published, stats.unavailable, stats.rejected) == (1, 1, 0)
    assert broker.publish.await_count == 2
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_invalid_row_is_skipped(settings) -> None:
    """Пустой текст не отправляется, остальные строки публикуются"""
    broker = fake_broker(None, None)

    stats = await Source(broker, settings).run(tasks(TEXTS[0], "   ", TEXTS[1]))

    assert (stats.published, stats.skipped, stats.next_row) == (2, 1, 3)


async def test_stop_interrupts_pause(settings, caplog) -> None:
    """Сигнал остановки прерывает паузу сразу, итог сообщает строку для продолжения"""
    slow = settings.model_copy(
        update={"source_initial_delay_seconds": 30, "source_max_delay_seconds": 30}
    )
    stop = asyncio.Event()
    broker = Mock(publish=AsyncMock(side_effect=queue_full()))
    asyncio.get_running_loop().call_later(0.1, stop.set)

    stats = await asyncio.wait_for(Source(broker, slow, stop).run(tasks(*TEXTS)), timeout=2)

    assert (stats.published, stats.next_row) == (0, 0)
    assert broker.publish.await_count == 1
    assert "source stopped" in caplog.messages


async def test_progress_report_contains_rate(settings, caplog) -> None:
    """Раз в интервал источник пишет скорость публикации и число отказов"""
    clock = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.0])
    reporting = settings.model_copy(update={"source_report_seconds": 1})
    broker = fake_broker(None, queue_full(), None, None)
    caplog.set_level(logging.INFO)

    await Source(broker, reporting, clock=lambda: next(clock)).run(tasks(*TEXTS[:3]))

    reports = [record for record in caplog.records if record.msg == "source progress"]
    assert [(r.published, r.rate, r.rejected_recently) for r in reports] == [
        (1, 1.0, 1),
        (3, 2.0, 0),
    ]


async def test_main_publishes_requested_rows(csv_file, monkeypatch) -> None:
    """Запуск из командной строки отправляет нужный диапазон строк в очередь задач"""
    broker = RabbitBroker()
    received = []

    @broker.subscriber(TASKS_QUEUE, TASKS_EXCHANGE)
    async def collect(task: TaskMessage) -> None:
        received.append(task.text)

    monkeypatch.setattr(source, "make_broker", lambda _: broker)
    async with TestRabbitBroker(broker):
        stats = await main(
            ["--file", str(csv_file), "--run", "cli", "--start", "1", "--count", "2"]
        )

    assert received == TEXTS[1:3]
    assert (stats.published, stats.next_row) == (2, 3)
