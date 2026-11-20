"""Воркер конвейера, берёт задачи из RabbitMQ и выполняет inference моделью из MLflow

Модель загружается один раз при старте, inference идёт в пуле потоков, поэтому цикл событий
свободен и воркер отвечает на проверку живости даже под нагрузкой
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import NoReturn, TypeVar

import asyncpg
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.exceptions import RejectMessage
from faststream.rabbit import Channel
from faststream.rabbit.annotations import RabbitMessage
from opentelemetry import trace
from pydantic import ValidationError

from src import tracing
from src.broker import TASKS_EXCHANGE, TASKS_QUEUE, declare_dead_letters, make_broker
from src.config import Settings, get_settings
from src.logging_config import setup_logging, task_context
from src.schemas import ProcessResponse, TaskMessage, TaskStatus
from src.services.inference import InferenceService, load_inference
from src.storage import complete_task, ensure_schema, fail_task, start_attempt

log = logging.getLogger(__name__)

T = TypeVar("T")
# Сбой связи с базой временный, такую задачу возвращают в очередь, а не в DLQ
STORAGE_ERRORS = (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)


def run_inference(
    inference: InferenceService, text: str, seconds: float, attempt: int = 1
) -> ProcessResponse:
    """Выполнить inference за заданное время, функция синхронная и работает в потоке пула"""
    with tracing.span(
        "inference",
        attributes={
            "model.alias": inference.info.alias,
            "model.version": inference.info.version,
            "delivery.attempt": attempt,
        },
    ):
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
            trace.get_current_span().set_attribute("delivery.attempt", 1)
            log.error(
                "malformed task moved to dead letter queue",
                extra={"error": exc.errors(include_input=False)[0]["msg"], "attempts": 1},
            )
            # Повтор не исправит некорректное сообщение, поэтому оно сразу уходит в DLQ
            raise RejectMessage(requeue=False) from None
        with task_context(task.task_id):
            await self._process(task)

    async def _process(self, task: TaskMessage) -> None:
        """Выполнить попытку и сохранить результат"""
        # Попытка учитывается до inference, поэтому считается и падение воркера посреди задачи
        attempt = await self._store(
            start_attempt(self.pool, task.task_id, task.text, task.enqueued_at)
        )
        current = trace.get_current_span()
        current.set_attribute("delivery.attempt", attempt.attempts)
        log.info("task received", extra={"attempts": attempt.attempts})
        if attempt.status is TaskStatus.DONE:
            log.info("duplicate task skipped")
            return
        if attempt.attempts > self.settings.task_max_deliveries:
            await self._give_up(task, attempt.attempts, "превышено число доставок")
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                copy_context().run,
                run_inference,
                self.inference,
                task.text,
                self.settings.inference_seconds,
                attempt.attempts,
            )
        except Exception as exc:
            log.exception("inference failed", extra={"attempts": attempt.attempts})
            if attempt.attempts >= self.settings.task_max_deliveries:
                await self._give_up(task, attempt.attempts, f"ошибка модели {type(exc).__name__}")
            await asyncio.sleep(self.settings.task_retry_delay_seconds)
            raise RejectMessage(requeue=True) from exc
        spans = [span.model_dump() for span in response.spans]
        with tracing.span(
            "database write",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system.name": "postgresql",
                "db.operation.name": "UPDATE",
                "delivery.attempt": attempt.attempts,
            },
        ):
            await self._store(
                complete_task(
                    self.pool, task.task_id, spans, response.model.name, response.model.version
                )
            )
        log.info(
            "task done",
            extra={"attempts": attempt.attempts, "spans": len(spans)},
        )

    async def _store(self, operation: Awaitable[T]) -> T:
        """Выполнить запрос к базе, при сбое связи вернуть задачу в очередь после паузы"""
        try:
            return await operation
        except STORAGE_ERRORS as exc:
            log.warning("storage unavailable, task requeued", extra={"error": type(exc).__name__})
            await asyncio.sleep(self.settings.task_retry_delay_seconds)
            raise RejectMessage(requeue=True) from exc

    async def _give_up(self, task: TaskMessage, attempts: int, reason: str) -> NoReturn:
        """Отметить задачу проваленной и отправить сообщение в DLQ без возврата в очередь"""
        await self._store(fail_task(self.pool, task.task_id, reason))
        log.error(
            "task moved to dead letter queue",
            extra={"attempts": attempts, "reason": reason},
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
    setup_logging(settings.log_level, "worker")
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
        received_at = time.time_ns()
        redelivered = bool(message.raw_message.redelivered)
        with (
            tracing.span(
                "process",
                kind=trace.SpanKind.CONSUMER,
                context=tracing.extract_context(message.headers),
                attributes={
                    "messaging.system": "rabbitmq",
                    "messaging.destination.name": TASKS_QUEUE.name,
                    "messaging.rabbitmq.message.redelivered": redelivered,
                },
            ) as span,
            task_context(message.correlation_id),
        ):
            published_at = message.headers.get("published_at_ns")
            if isinstance(published_at, int) and published_at > 0 and not redelivered:
                span.set_attribute("queue.wait.seconds", max(0, received_at - published_at) / 1e9)
            batch = message.headers.get("batch")
            if isinstance(batch, str) and batch:
                span.set_attribute("batch.name", batch)
            try:
                await state.processor.handle(message.body)
            except RejectMessage as exc:
                span.add_event("requeue" if exc.extra_options.get("requeue") else "dlq")
                raise
            except Exception:
                span.add_event("dlq")
                raise

    async def startup() -> None:
        """Один раз загрузить модель и подключиться к базе до начала приёма задач"""
        tracing.configure("worker")
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

    async def stopping() -> None:
        log.info("worker stopping")

    async def shutdown() -> None:
        """После остановки брокера дождаться потоков пула и закрыть соединения с базой"""
        await asyncio.get_running_loop().run_in_executor(None, executor.shutdown)
        await state.processor.pool.close()
        log.info("worker stopped")

    return AsgiFastStream(
        broker,
        asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=5.0))],
        on_startup=[startup],
        on_shutdown=[stopping],
        after_shutdown=[shutdown],
        logger=log,
    )


app = create_worker(get_settings())
