"""Общие фикстуры тестов"""

import os
import re
from collections.abc import AsyncIterator, Iterator
from unittest.mock import Mock

# Окружение задаётся до импорта src.app, потому что приложение собирается при импорте
# Порт 59999 заведомо закрыт, поэтому сценарий с недоступной БД повторяется на любой машине
os.environ["POSTGRES_HOST"] = "127.0.0.1"
os.environ["POSTGRES_PORT"] = "59999"
os.environ["POSTGRES_USER"] = "test"
os.environ["POSTGRES_PASSWORD"] = "test"
os.environ["POSTGRES_DB"] = "test"
os.environ["HEALTH_TIMEOUT"] = "1"
os.environ["RABBITMQ_HOST"] = "127.0.0.1"
os.environ["RABBITMQ_USER"] = "test"
os.environ["RABBITMQ_PASSWORD"] = "test"
os.environ["TASKS_QUEUE_LIMIT"] = "100"

import asyncio

import aio_pika
import asyncpg
import httpx
import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from src.app import app
from src.broker import DEAD_LETTER_QUEUE, QUEUE_POLICY_NAME, TASKS_QUEUE, apply_queue_policy
from src.schemas import ModelInfo
from src.services.inference import InferenceService
from src.storage import ensure_schema


@pytest.fixture(scope="session")
def postgres_url():
    """Настоящий Postgres в контейнере, один на весь прогон тестов"""
    with PostgresContainer("postgres:17-alpine", driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def rabbitmq_settings() -> Iterator[dict]:
    """Настоящий RabbitMQ в контейнере, та же версия, что в compose, один на весь прогон"""
    # Логин и пароль совпадают с окружением выше, guest RabbitMQ пускает только с localhost
    container = RabbitMqContainer(
        "rabbitmq:4.4-management-alpine", username="test", password="test"
    ).with_exposed_ports(15672)
    with container:
        yield {
            "rabbitmq_host": container.get_container_host_ip(),
            "rabbitmq_port": int(container.get_exposed_port(5672)),
            "rabbitmq_management_port": int(container.get_exposed_port(15672)),
        }


@pytest.fixture
async def rabbit_connection(settings) -> AsyncIterator[aio_pika.abc.AbstractConnection]:
    """Соединение с брокером из контейнера для проверки очереди и тестовых потребителей

    Адрес берётся из фикстуры settings того модуля, где идёт тест
    """
    connection = await aio_pika.connect(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        login=settings.rabbitmq_user,
        password=settings.rabbitmq_password.get_secret_value(),
    )
    yield connection
    await connection.close()


@pytest.fixture
async def limited_tasks_queue(settings, rabbit_connection) -> AsyncIterator[None]:
    """Очередь задач с лимитом из settings теста, после теста политика и очереди удаляются"""
    channel = await rabbit_connection.channel()
    for name in (TASKS_QUEUE.name, DEAD_LETTER_QUEUE.name):
        await channel.queue_delete(name)
    base = f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}"
    auth = (settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value())
    async with httpx.AsyncClient(timeout=5) as client:
        # Management API поднимается позже AMQP, поэтому первые попытки могут не пройти
        for _ in range(60):
            try:
                await apply_queue_policy(settings, client)
                break
            except httpx.HTTPError:
                await asyncio.sleep(0.5)
        else:
            pytest.fail("Management API не ответил")
        yield
        await client.delete(f"{base}/api/policies/%2F/{QUEUE_POLICY_NAME}", auth=auth)
    for name in (TASKS_QUEUE.name, DEAD_LETTER_QUEUE.name):
        await channel.queue_delete(name)


@pytest.fixture
async def task_pool(postgres_url) -> AsyncIterator[asyncpg.Pool]:
    """Пул к Postgres с пустой таблицей задач перед каждым тестом"""
    pool = await asyncpg.create_pool(postgres_url, min_size=1, max_size=10)
    await ensure_schema(pool)
    await pool.execute("TRUNCATE tasks")
    yield pool
    await pool.close()


@pytest.fixture
def fake_inference() -> InferenceService:
    """Сервис inference с моделью, которая находит слово idiot без обращения к MLflow"""
    model = Mock()

    def predict(frame):
        """Вернуть позиции известного слова для проверки контракта API"""
        return pd.DataFrame(
            {
                "positions": [
                    [
                        pos
                        for match in re.finditer("idiot", text, re.IGNORECASE)
                        for pos in range(match.start(), match.end())
                    ]
                    for text in frame["text"]
                ]
            }
        )

    model.predict.side_effect = predict
    return InferenceService(
        model=model,
        info=ModelInfo(
            name="toxic-spans-detector", version="1", alias="champion", run_id="run-one"
        ),
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """HTTP клиент приложения, lifespan создаёт пул БД, а брокер работает в памяти"""
    async with TestRabbitBroker(app.state.broker, connect_only=True), LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
