"""Воркер конвейера, берёт задачи из RabbitMQ и выполняет inference моделью из MLflow

Модель загружается один раз при старте, inference идёт в пуле потоков, поэтому цикл событий
свободен и воркер отвечает на проверку живости даже под нагрузкой
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NoReturn, TypeVar

import asyncpg
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.exceptions import RejectMessage
from faststream.rabbit import Channel
from faststream.rabbit.annotations import RabbitMessage
from pydantic import ValidationError

from src.broker import TASKS_EXCHANGE, TASKS_QUEUE, declare_dead_letters, make_broker
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.schemas import ProcessResponse, TaskMessage, TaskStatus
from src.services.inference import InferenceService, load_inference
from src.storage import complete_task, ensure_schema, fail_task, start_attempt

log = logging.getLogger(__name__)

T = TypeVar("T")
# Сбой связи с базой временный, такую задачу возвращают в очередь, а не в DLQ
STORAGE_ERRORS = (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)


def run_inference(inference: InferenceService, text: str, seconds: float) -> ProcessResponse:
    """Выполнить inference за заданное время, функция синхронная и работает в потоке пула"""
    started = time.perf_counter()
    response = inference.process(text)
    remaining = seconds - (time.perf_counter() - started)
    if remaining > 0:
        # Пауза имитирует тяжёлую модель и идёт в потоке пула, цикл событий она не блокирует
        time.sleep(remaining)
    return response


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Создать пул соединений воркера, одной задаче нужно одно соединение за раз"""
    return await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        database=settings.postgres_db,
        min_size=1,
        max_size=settings.worker_concurrency,
    )


@dataclass
class TaskProcessor:
    """Обработка одной задачи, исход определяет, подтвердить сообщение или отклонить"""

    settings: Settings
    pool: asyncpg.Pool
    inference: InferenceService
    executor: ThreadPoolExecutor

    async def handle(self, body: bytes) -> None:
        """Выполнить задачу и вернуть управление для ack или бросить RejectMessage"""
        try:
            task = TaskMessage.model_validate_json(body)
        except ValidationError as exc:
            log.warning("malformed task rejected", extra={"error": str(exc.errors()[0]["msg"])})
            # Повтор не исправит некорректное сообщение, поэтому оно сразу уходит в DLQ
            raise RejectMessage(requeue=False) from exc
        task_id = str(task.task_id)
        # Попытка учитывается до inference, поэтому считается и падение воркера посреди задачи
        attempt = await self._store(
            start_attempt(self.pool, task.task_id, task.text, task.enqueued_at)
        )
        if attempt.status is TaskStatus.DONE:
            log.info("duplicate task skipped", extra={"task_id": task_id})
            return
        if attempt.attempts > self.settings.task_max_deliveries:
            await self._give_up(task, attempt.attempts, "превышено число доставок")
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                run_inference,
                self.inference,
                task.text,
                self.settings.inference_seconds,
            )
        except Exception as exc:
            log.exception(
                "inference failed", extra={"task_id": task_id, "attempts": attempt.attempts}
            )
            if attempt.attempts >= self.settings.task_max_deliveries:
                await self._give_up(task, attempt.attempts, f"ошибка модели {type(exc).__name__}")
            await asyncio.sleep(self.settings.task_retry_delay_seconds)
            raise RejectMessage(requeue=True) from exc
        spans = [span.model_dump() for span in response.spans]
        await self._store(
            complete_task(
                self.pool, task.task_id, spans, response.model.name, response.model.version
            )
        )
        log.info(
            "task done",
            extra={"task_id": task_id, "attempts": attempt.attempts, "spans": len(spans)},
        )

    async def _store(self, operation: Awaitable[T]) -> T:
        """Выполнить запрос к базе, при сбое связи вернуть задачу в очередь после паузы"""
        try:
            return await operation
        except STORAGE_ERRORS as exc:
            log.warning("storage unavailable, task requeued", extra={"error": repr(exc)})
            await asyncio.sleep(self.settings.task_retry_delay_seconds)
            raise RejectMessage(requeue=True) from exc

    async def _give_up(self, task: TaskMessage, attempts: int, reason: str) -> NoReturn:
        """Отметить задачу проваленной и отправить сообщение в DLQ без возврата в очередь"""
        await self._store(fail_task(self.pool, task.task_id, reason))
        log.error(
            "task moved to dead letter queue",
            extra={"task_id": str(task.task_id), "attempts": attempts, "reason": reason},
        )
        raise RejectMessage(requeue=False)


@dataclass
class WorkerState:
    """Ресурсы, которые появляются при старте воркера"""

    processor: TaskProcessor | None = None


def create_worker(
    settings: Settings,
    load_model: Callable[[Settings], InferenceService] = load_inference,
    connect: Callable[[Settings], Awaitable[asyncpg.Pool]] = create_pool,
) -> AsgiFastStream:
    """Собрать воркер, загрузку модели и подключение к базе можно подменить в тестах"""
    setup_logging(settings.log_level)
    broker = make_broker(settings, graceful_timeout=settings.worker_graceful_timeout)
    executor = ThreadPoolExecutor(
        max_workers=settings.worker_concurrency, thread_name_prefix="inference"
    )
    state = WorkerState()

    @broker.subscriber(
        TASKS_QUEUE,
        TASKS_EXCHANGE,
        # Обработчик получает не больше сообщений, чем потоков в пуле, лишние ждут в очереди
        channel=Channel(prefetch_count=settings.worker_prefetch_count),
        # Подтверждение уходит после записи результата, неожиданная ошибка отправляет
        # сообщение в DLQ, а не в бесконечный повтор
        ack_policy=AckPolicy.REJECT_ON_ERROR,
    )
    async def handle_task(message: RabbitMessage) -> None:
        """Передать сообщение обработчику задач"""
        await state.processor.handle(message.body)

    async def startup() -> None:
        """Один раз загрузить модель и подключиться к базе до начала приёма задач"""
        inference = await asyncio.get_running_loop().run_in_executor(None, load_model, settings)
        pool = await connect(settings)
        await ensure_schema(pool)
        # Без обменника недоставленных сообщений отклонённая задача пропала бы, поэтому воркер
        # объявляет его сам и не зависит от того, запускалась ли настройка брокера
        # Очередь задач объявляет подписчик на своём канале с prefetch, FastStream запоминает
        # объявленную очередь, и объявление здесь привязало бы потребителя к каналу без лимита
        await broker.connect()
        await declare_dead_letters(broker)
        state.processor = TaskProcessor(settings, pool, inference, executor)
        log.info(
            "worker started",
            extra={
                "model_version": inference.info.version,
                "model_alias": inference.info.alias,
                "concurrency": settings.worker_concurrency,
                "prefetch_count": settings.worker_prefetch_count,
                "inference_seconds": settings.inference_seconds,
            },
        )

    async def shutdown() -> None:
        """После остановки брокера дождаться потоков пула и закрыть соединения с базой"""
        await asyncio.get_running_loop().run_in_executor(None, executor.shutdown)
        await state.processor.pool.close()
        log.info("worker stopped")

    return AsgiFastStream(
        broker,
        asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=5.0))],
        on_startup=[startup],
        after_shutdown=[shutdown],
        logger=log,
    )


app = create_worker(get_settings())
