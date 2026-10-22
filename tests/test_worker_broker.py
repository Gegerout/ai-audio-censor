"""Воркер на настоящем RabbitMQ в контейнере, брокер в памяти не умеет prefetch и DLQ"""

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import aio_pika
import asyncpg
import pandas as pd
import pytest
from faststream import TestApp
from faststream.asgi import AsgiFastStream

from src.broker import DEAD_LETTER_QUEUE, TASKS_EXCHANGE, TASKS_QUEUE, TASKS_ROUTING_KEY
from src.config import Settings, get_settings
from src.schemas import TaskMessage, TaskStatus
from src.storage import get_task
from src.worker import create_worker


@pytest.fixture
def settings(rabbitmq_settings) -> Settings:
    """Настройки воркера с адресом брокера из контейнера"""
    return get_settings().model_copy(
        update={**rabbitmq_settings, "inference_seconds": 0, "task_retry_delay_seconds": 0}
    )


@pytest.fixture
async def channel(settings) -> AsyncIterator[aio_pika.abc.AbstractChannel]:
    """Канал для проверки очередей, перед тестом очереди удаляются, их объявит воркер"""
    connection = await aio_pika.connect(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        login=settings.rabbitmq_user,
        password=settings.rabbitmq_password.get_secret_value(),
    )
    channel = await connection.channel()
    for name in (TASKS_QUEUE.name, DEAD_LETTER_QUEUE.name):
        await channel.queue_delete(name)
    yield channel
    await connection.close()


@pytest.fixture
def make_worker(settings, postgres_url, fake_inference, task_pool) -> Callable[..., AsgiFastStream]:
    """Собрать воркер с моделью без MLflow и пулом к тестовому Postgres"""

    def build(**overrides) -> AsgiFastStream:
        return create_worker(
            settings.model_copy(update=overrides),
            load_model=Mock(return_value=fake_inference),
            connect=lambda _: asyncpg.create_pool(postgres_url, min_size=1, max_size=4),
        )

    return build


async def ready(channel: aio_pika.abc.AbstractChannel, name: str) -> int:
    """Число сообщений, которые ждут в очереди и ещё не отданы потребителю"""
    queue = await channel.declare_queue(name, passive=True)
    return queue.declaration_result.message_count


async def wait_until(check: Callable[[], Awaitable[bool]]) -> None:
    """Дождаться условия в брокере или базе, если за пять секунд не дождались, тест падает"""
    for _ in range(100):
        if await check():
            return
        await asyncio.sleep(0.05)
    pytest.fail("условие не выполнилось за пять секунд")


async def publish(app: AsgiFastStream, message: TaskMessage | bytes) -> None:
    """Опубликовать сообщение с подтверждением брокера"""
    await app.broker.publish(
        message, exchange=TASKS_EXCHANGE, routing_key=TASKS_ROUTING_KEY, persist=True
    )


def task() -> TaskMessage:
    """Создать задачу с новым id"""
    return TaskMessage(task_id=uuid4(), text="You are an idiot.", enqueued_at=datetime.now(UTC))


async def test_worker_takes_only_prefetch_messages(
    make_worker, fake_inference, task_pool, channel
) -> None:
    """Воркер берёт из очереди не больше prefetch, остальные задачи ждут в очереди брокера"""
    release = threading.Event()

    def predict(frame):
        release.wait(timeout=5)
        return pd.DataFrame({"positions": [[]]})

    fake_inference.model.predict.side_effect = predict
    app = make_worker(worker_concurrency=2)

    async def taken() -> int:
        return await task_pool.fetchval("SELECT count(*) FROM tasks")

    async with TestApp(app):
        try:
            for _ in range(6):
                await publish(app, task())
            await wait_until(lambda: _equals(taken, 2))
            await asyncio.sleep(0.3)
            assert await taken() == 2
            assert await ready(channel, TASKS_QUEUE.name) == 4
        finally:
            release.set()
        await wait_until(lambda: _done_count(task_pool, 6))

    assert await task_pool.fetchval("SELECT max(attempts) FROM tasks") == 1


async def test_malformed_message_reaches_dead_letter_queue(make_worker, channel) -> None:
    """Некорректное сообщение попадает в DLQ, которую воркер объявил сам"""
    app = make_worker()

    async with TestApp(app):
        await publish(app, b"not json")

        await wait_until(lambda: _ready_equals(channel, DEAD_LETTER_QUEUE.name, 1))
    assert await ready(channel, TASKS_QUEUE.name) == 0


async def test_failing_task_is_redelivered_then_dead_lettered(
    make_worker, fake_inference, task_pool, channel
) -> None:
    """Брокер повторно доставляет задачу до лимита, затем она уходит в DLQ и не крутится дальше"""

    def failing_predict(frame):
        raise RuntimeError("model crashed")

    fake_inference.model.predict.side_effect = failing_predict
    app = make_worker(task_max_deliveries=3)
    message = task()

    async with TestApp(app):
        await publish(app, message)

        await wait_until(lambda: _ready_equals(channel, DEAD_LETTER_QUEUE.name, 1))
        await asyncio.sleep(0.3)

    record = await get_task(task_pool, message.task_id)
    assert record.status is TaskStatus.FAILED
    assert record.attempts == 3
    assert fake_inference.model.predict.call_count == 3
    assert await ready(channel, TASKS_QUEUE.name) == 0


async def _equals(count: Callable[[], Awaitable[int]], expected: int) -> bool:
    return await count() == expected


async def _ready_equals(channel, name: str, expected: int) -> bool:
    return await ready(channel, name) == expected


async def _done_count(pool: asyncpg.Pool, expected: int) -> bool:
    done = await pool.fetchval("SELECT count(*) FROM tasks WHERE status = 'done'")
    return done == expected
