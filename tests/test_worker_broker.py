"""Воркер на настоящем RabbitMQ в контейнере, брокер в памяти не умеет prefetch и DLQ"""

import asyncio
import json
import logging
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
from opentelemetry.trace import StatusCode, format_span_id, format_trace_id

from src.broker import (
    DEAD_LETTER_QUEUE,
    TASKS_EXCHANGE,
    TASKS_QUEUE,
    TASKS_ROUTING_KEY,
    publish_message,
)
from src.config import Settings, get_settings
from src.logging_config import JsonFormatter
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


@pytest.mark.parametrize(
    "body",
    [b"not json", task().model_copy(update={"text": "a\x00b"}).model_dump_json().encode()],
    ids=["invalid-json", "nul-in-text"],
)
async def test_malformed_message_reaches_dead_letter_queue(make_worker, channel, body) -> None:
    """Некорректное сообщение попадает в DLQ, которую воркер объявил сам"""
    app = make_worker()

    async with TestApp(app):
        await publish(app, body)

        await wait_until(lambda: _ready_equals(channel, DEAD_LETTER_QUEUE.name, 1))
    assert await ready(channel, TASKS_QUEUE.name) == 0


async def test_failing_task_is_redelivered_then_dead_lettered(
    make_worker, fake_inference, task_pool, channel, caplog, spans
) -> None:
    """Брокер повторно доставляет задачу до лимита, затем она уходит в DLQ и не крутится дальше"""

    def failing_predict(frame):
        raise RuntimeError(frame.iloc[0]["text"])

    fake_inference.model.predict.side_effect = failing_predict
    app = make_worker(task_max_deliveries=3)
    caplog.handler.setFormatter(JsonFormatter("worker"))
    logging.getLogger().addHandler(caplog.handler)
    message = task()

    async with TestApp(app):
        await publish_message(app.broker, message)

        await wait_until(lambda: _ready_equals(channel, DEAD_LETTER_QUEUE.name, 1))
        await asyncio.sleep(0.3)

    finished = spans.get_finished_spans()
    (producer,) = [s for s in finished if s.name == "publish"]
    deliveries = [s for s in finished if s.name == "process"]
    assert [s.attributes["delivery.attempt"] for s in deliveries] == [1, 2, 3]
    assert [s.attributes["messaging.rabbitmq.message.redelivered"] for s in deliveries] == [
        False,
        True,
        True,
    ]
    assert all(
        s.parent.span_id == producer.context.span_id and s.status.status_code is StatusCode.ERROR
        for s in deliveries
    )
    assert all(s.context.trace_id == producer.context.trace_id for s in finished)
    for delivery, outcome in zip(deliveries, ("requeue", "requeue", "dlq"), strict=True):
        assert {e.name for e in delivery.events} == {"exception", outcome}
    assert all(message.text not in s.to_json() for s in finished)
    record = await get_task(task_pool, message.task_id)
    assert record.status is TaskStatus.FAILED
    assert record.attempts == 3
    assert fake_inference.model.predict.call_count == 3
    events = [json.loads(line) for line in caplog.text.splitlines()]
    failures = [e for e in events if e["msg"] == "inference failed"]
    assert [e["attempts"] for e in failures] == [1, 2, 3]
    assert all(e.get("task_id") == str(message.task_id) for e in failures)
    assert all(e["level"] == "ERROR" and "failing_predict" in e["exc_info"] for e in failures)
    assert message.text not in caplog.text
    assert await ready(channel, TASKS_QUEUE.name) == 0


async def _equals(count: Callable[[], Awaitable[int]], expected: int) -> bool:
    return await count() == expected


async def _ready_equals(channel, name: str, expected: int) -> bool:
    return await ready(channel, name) == expected


async def _done_count(pool: asyncpg.Pool, expected: int) -> bool:
    done = await pool.fetchval("SELECT count(*) FROM tasks WHERE status = 'done'")
    return done == expected


async def test_worker_correlates_concurrent_model_and_broker_logs(
    make_worker, fake_inference, task_pool, channel, caplog, spans
) -> None:
    """Записи FastStream и потоков модели сохраняют id каждой доставки"""
    messages = [task(), task()]
    predict = fake_inference.model.predict.side_effect

    def logged_predict(frame):
        logging.getLogger("model").info("predict running")
        return predict(frame)

    fake_inference.model.predict.side_effect = logged_predict
    caplog.set_level(logging.DEBUG)
    app = make_worker(log_level="DEBUG")
    caplog.handler.setFormatter(JsonFormatter("worker"))
    logging.getLogger().addHandler(caplog.handler)
    async with TestApp(app):
        await asyncio.gather(
            *(publish_message(app.broker, message, "test-batch") for message in messages)
        )
        invalid_id = str(uuid4())
        await app.broker.publish(
            b"private invalid payload",
            exchange=TASKS_EXCHANGE,
            routing_key=TASKS_ROUTING_KEY,
            correlation_id=invalid_id,
        )

        await wait_until(lambda: _done_count(task_pool, len(messages)))
        await wait_until(lambda: _ready_equals(channel, DEAD_LETTER_QUEUE.name, 1))

    output = caplog.text
    events = [json.loads(line) for line in output.splitlines()]
    for message in messages:
        names = [e["msg"] for e in events if e.get("task_id") == str(message.task_id)]
        assert names == ["Received", "task received", "predict running", "task done", "Processed"]
        (process,) = [
            s
            for s in spans.get_finished_spans()
            if s.name == "process" and s.attributes["task.id"] == str(message.task_id)
        ]
        trace_id = process.context.trace_id
        tree = {s.name: s for s in spans.get_finished_spans() if s.context.trace_id == trace_id}
        assert process.parent.span_id == tree["publish"].context.span_id
        assert process.attributes["batch.name"] == "test-batch"
        assert process.attributes["queue.wait.seconds"] >= 0
        for name in ("inference", "database write"):
            assert tree[name].parent.span_id == process.context.span_id
        inference = tree["inference"]
        assert (inference.attributes["model.alias"], inference.attributes["model.version"]) == (
            fake_inference.info.alias,
            fake_inference.info.version,
        )
        (entry,) = [
            e
            for e in events
            if e["msg"] == "predict running" and e["task_id"] == str(message.task_id)
        ]
        assert (entry["trace_id"], entry["span_id"]) == (
            format_trace_id(trace_id),
            format_span_id(inference.context.span_id),
        )
        assert message.text not in output
    assert any(e["level"] == "ERROR" and e.get("task_id") == invalid_id for e in events)
    assert "private invalid payload" not in output
    assert all("task_id" not in e for e in events if e["msg"].startswith("worker "))
