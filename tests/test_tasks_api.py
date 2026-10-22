"""Тесты API задач на настоящем Postgres и брокере в памяти"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import asyncpg
import pytest
from aiormq.exceptions import (
    AMQPConnectionError,
    ChannelInvalidStateError,
    DeliveryError,
    PublishError,
)
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from pamqp.commands import Basic
from pydantic import SecretStr

from src.app import create_app
from src.broker import TASKS_EXCHANGE, TASKS_QUEUE
from src.config import Settings, get_settings
from src.schemas import TaskMessage
from src.services import tasks
from src.services.tasks import estimate_retry_after, retry_after_seconds
from src.storage import complete_task, create_pending, start_attempt


@dataclass
class Api:
    """Клиент приложения и то, что приложение опубликовало в брокер"""

    http: AsyncClient
    app: FastAPI
    published: list[TaskMessage]


@pytest.fixture
def settings(postgres_url) -> Settings:
    """Настройки с адресом тестового Postgres"""
    url = urlsplit(postgres_url)
    return get_settings().model_copy(
        update={
            "postgres_host": url.hostname,
            "postgres_port": url.port,
            "postgres_user": url.username,
            "postgres_password": SecretStr(url.password),
            "postgres_db": url.path.lstrip("/"),
        }
    )


@pytest.fixture
async def api(settings, task_pool) -> AsyncIterator[Api]:
    """Отдельное приложение на тестовом Postgres, публикации попадают в список"""
    app = create_app(settings)
    published = []

    @app.state.broker.subscriber(TASKS_QUEUE, TASKS_EXCHANGE)
    async def collect(task: TaskMessage) -> None:
        published.append(task)

    # Брокер запускается целиком, иначе подписчик сборщика не будет готов
    async with TestRabbitBroker(app.state.broker, connect_only=False), LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            yield Api(http, app, published)


def reject_with(monkeypatch, api: Api, error: Exception) -> None:
    """Брокер отвечает на публикацию заданной ошибкой"""
    monkeypatch.setattr(api.app.state.broker, "publish", AsyncMock(side_effect=error))


def queue_full() -> DeliveryError:
    """Отказ брокера при заполненной очереди"""
    return DeliveryError(None, Basic.Nack(delivery_tag=1))


async def stored(pool: asyncpg.Pool) -> int:
    """Сколько задач в базе"""
    return await pool.fetchval("SELECT count(*) FROM tasks")


async def test_task_is_accepted_and_queued(api, task_pool) -> None:
    """Задача записывается в базу как pending, уходит в очередь и получает адрес результата"""
    response = await api.http.post("/api/v1/tasks", json={"text": "You are an idiot."})

    assert response.status_code == 202
    body = response.json()
    task_id = UUID(body["task_id"])
    assert body["status"] == "pending"
    assert response.headers["Location"] == f"/api/v1/tasks/{task_id}"
    assert [(task.task_id, task.text) for task in api.published] == [(task_id, "You are an idiot.")]
    assert await task_pool.fetchval("SELECT status FROM tasks WHERE id = $1", task_id) == "pending"


async def test_task_result_is_returned_when_done(api, task_pool) -> None:
    """Пока воркер не закончил, задача pending, потом приходят фрагменты и версия модели"""
    created = await api.http.post("/api/v1/tasks", json={"text": "You are an idiot."})
    location = created.headers["Location"]
    task_id = UUID(created.json()["task_id"])

    pending = (await api.http.get(location)).json()
    message = api.published[0]
    await start_attempt(task_pool, task_id, message.text, message.enqueued_at)
    spans = [{"start": 11, "end": 16, "text": "idiot"}]
    await complete_task(task_pool, task_id, spans, "toxic-spans-detector", "1")
    done = await api.http.get(location)

    assert pending["status"] == "pending"
    assert pending["spans"] is None
    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "done"
    assert body["spans"] == spans
    assert (body["model_name"], body["model_version"], body["attempts"]) == (
        "toxic-spans-detector",
        "1",
        1,
    )
    assert body["finished_at"] is not None


async def test_unknown_task_is_not_found(api) -> None:
    """Несуществующая задача даёт 404, неверный id даёт 422"""
    missing = await api.http.get(f"/api/v1/tasks/{uuid4()}")
    invalid = await api.http.get("/api/v1/tasks/not-a-uuid")

    assert missing.status_code == 404
    assert invalid.status_code == 422


@pytest.mark.parametrize(
    "body", [{}, {"text": ""}, {"text": " \n "}, {"text": 123}, {"text": "x" * 10001}]
)
async def test_invalid_text_is_rejected_before_queue(api, task_pool, body) -> None:
    """Некорректный текст отклоняется с 422 и не попадает ни в базу, ни в очередь"""
    response = await api.http.post("/api/v1/tasks", json=body)

    assert response.status_code == 422
    assert api.published == []
    assert await stored(task_pool) == 0


async def test_full_queue_gives_503_at_once_without_pending(api, task_pool, monkeypatch) -> None:
    """Заполненная очередь сразу даёт 503 со временем повтора, строка в базе откатывается"""
    reject_with(monkeypatch, api, queue_full())
    monkeypatch.setattr(tasks, "queue_load", AsyncMock(return_value=(100, 1)))

    started = time.perf_counter()
    response = await api.http.post("/api/v1/tasks", json={"text": "You are an idiot."})
    elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json() == {"detail": "Очередь задач заполнена, повторите позже"}
    assert elapsed < 0.5
    assert await stored(task_pool) == 0


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (ChannelInvalidStateError("closed"), "Брокер недоступен"),
        (AMQPConnectionError("closed"), "Брокер недоступен"),
        (
            PublishError(Mock(delivery=Basic.Return(312, "NO_ROUTE", "tasks", "tasks")), None),
            "Очередь задач не объявлена",
        ),
    ],
)
async def test_broker_problem_gives_503_without_pending(
    api, task_pool, monkeypatch, error, detail
) -> None:
    """Без брокера задача не принимается, клиент получает верхнюю границу времени повтора"""
    reject_with(monkeypatch, api, error)

    response = await api.http.post("/api/v1/tasks", json={"text": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == detail
    assert response.headers["Retry-After"] == "30"
    assert await stored(task_pool) == 0


async def test_hanging_broker_does_not_hold_request(api, task_pool, monkeypatch) -> None:
    """Зависший брокер не держит запрос дольше таймаута, и строка тоже откатывается"""
    api.app.state.settings = api.app.state.settings.model_copy(
        update={"tasks_timeout_seconds": 0.1}
    )

    async def hang(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(api.app.state.broker, "publish", hang)
    started = time.perf_counter()
    response = await api.http.post("/api/v1/tasks", json={"text": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "База или брокер не ответили вовремя"
    assert time.perf_counter() - started < 1
    assert await stored(task_pool) == 0


async def test_unavailable_database_gives_503(client) -> None:
    """Если база недоступна, задача не принимается, а чтение результата тоже даёт 503"""
    created = await client.post("/api/v1/tasks", json={"text": "hello"})
    read = await client.get(f"/api/v1/tasks/{uuid4()}")

    assert created.status_code == 503
    assert created.json()["detail"] == "База недоступна"
    assert read.status_code == 503


async def test_old_process_endpoint_is_gone(client) -> None:
    """Синхронный POST /process заменён на очередь задач"""
    response = await client.post("/process", json={"text": "hello"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("messages", "consumers", "expected"),
    [(100, 1, 5), (100, 2, 3), (1, 1, 1), (0, 1, 1), (100, 0, 30), (10_000, 1, 30)],
)
def test_retry_after_is_time_to_drain_queue(messages, consumers, expected) -> None:
    """Время повтора равно времени, за которое воркеры разберут очередь, от 1 до 30 секунд"""
    settings = get_settings().model_copy(update={"worker_concurrency": 4, "inference_seconds": 0.2})

    assert retry_after_seconds(messages, consumers, settings) == expected


def test_retry_after_without_inference_delay() -> None:
    """Если inference мгновенный, клиент может повторить через секунду"""
    settings = get_settings().model_copy(update={"inference_seconds": 0})

    assert retry_after_seconds(100, 1, settings) == 1


async def test_estimate_falls_back_to_cap_when_broker_fails(monkeypatch) -> None:
    """Если брокер не ответил на запрос глубины очереди, берётся верхняя граница"""
    monkeypatch.setattr(tasks, "queue_load", AsyncMock(side_effect=AMQPConnectionError("down")))

    assert await estimate_retry_after(Mock(), get_settings()) == 30


async def test_worker_waits_for_api_transaction(task_pool) -> None:
    """Воркер, получивший сообщение до фиксации API, ждёт транзакцию и не создаёт вторую строку"""
    task_id = uuid4()
    enqueued = datetime.now(UTC)
    async with task_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        await create_pending(connection, task_id, "hello", enqueued)
        attempt = asyncio.create_task(start_attempt(task_pool, task_id, "hello", enqueued))
        await asyncio.sleep(0.3)
        blocked = not attempt.done()
        await transaction.commit()
        result = await asyncio.wait_for(attempt, timeout=2)

    assert blocked
    assert (result.status, result.attempts) == ("processing", 1)
    assert await stored(task_pool) == 1
