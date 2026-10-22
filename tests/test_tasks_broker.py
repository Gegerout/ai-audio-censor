"""API задач на настоящем RabbitMQ с ограниченной очередью и настоящем Postgres"""

import asyncio
import time
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import aio_pika
import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.app import create_app
from src.broker import TASKS_QUEUE
from src.config import Settings, get_settings

LIMIT = 3


@pytest.fixture
def settings(rabbitmq_settings, postgres_url) -> Settings:
    """Настройки API с брокером и базой из контейнеров и маленьким лимитом очереди

    Один поток и две секунды на задачу дают воркеру скорость ползадачи в секунду,
    так Retry-After легко проверить по числу воркеров
    """
    url = urlsplit(postgres_url)
    return get_settings().model_copy(
        update={
            **rabbitmq_settings,
            "postgres_host": url.hostname,
            "postgres_port": url.port,
            "postgres_user": url.username,
            "postgres_password": SecretStr(url.password),
            "postgres_db": url.path.lstrip("/"),
            "tasks_queue_limit": LIMIT,
            "worker_concurrency": 1,
            "inference_seconds": 2,
        }
    )


@pytest.fixture
async def http(settings, limited_tasks_queue, task_pool) -> AsyncIterator[AsyncClient]:
    """Клиент API, которое подключено к настоящему брокеру"""
    app = create_app(settings)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


async def post(http: AsyncClient) -> httpx.Response:
    """Отправить задачу"""
    return await http.post("/api/v1/tasks", json={"text": "You are an idiot."})


async def hold_one(connection: aio_pika.abc.AbstractConnection) -> None:
    """Потребитель, который забирает одно сообщение и не подтверждает его, как занятый воркер"""
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.get_queue(TASKS_QUEUE.name)
    taken = asyncio.Event()

    async def keep(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        taken.set()

    await queue.consume(keep)
    await asyncio.wait_for(taken.wait(), timeout=5)


async def test_full_queue_rejects_task_at_once(http, task_pool) -> None:
    """Брокер отказывает при заполненной очереди, API сразу отвечает 503 без строки pending"""
    accepted = [await post(http) for _ in range(LIMIT)]
    started = time.perf_counter()
    rejected = await post(http)
    elapsed = time.perf_counter() - started

    assert [response.status_code for response in accepted] == [202] * LIMIT
    assert rejected.status_code == 503
    assert rejected.json()["detail"] == "Очередь задач заполнена, повторите позже"
    # Воркеров нет, очередь не разбирается, поэтому клиент получает верхнюю границу
    assert rejected.headers["Retry-After"] == "30"
    assert elapsed < 1
    statuses = await task_pool.fetch("SELECT status, count(*) FROM tasks GROUP BY status")
    assert [tuple(row) for row in statuses] == [("pending", LIMIT)]


async def test_retry_after_shrinks_with_more_workers(http, rabbit_connection) -> None:
    """Retry-After равен времени разбора очереди и уменьшается, когда воркеров больше"""
    for _ in range(LIMIT):
        await post(http)
    await hold_one(rabbit_connection)
    await post(http)
    one_worker = await post(http)
    await hold_one(rabbit_connection)
    await post(http)
    two_workers = await post(http)

    # Три задачи в очереди, один воркер разбирает ползадачи в секунду, это 6 секунд
    assert one_worker.status_code == 503
    assert one_worker.headers["Retry-After"] == "6"
    # Два воркера разбирают те же три задачи вдвое быстрее
    assert two_workers.status_code == 503
    assert two_workers.headers["Retry-After"] == "3"
