"""Общие фикстуры тестов"""

import os
import re
from collections.abc import AsyncIterator
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

import asyncpg
import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from testcontainers.community.postgres import PostgresContainer

from src.app import app
from src.schemas import ModelInfo
from src.services.inference import InferenceService
from src.storage import ensure_schema


@pytest.fixture(scope="session")
def postgres_url():
    """Настоящий Postgres в контейнере, один на весь прогон тестов"""
    with PostgresContainer("postgres:17-alpine", driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
async def task_pool(postgres_url) -> AsyncIterator[asyncpg.Pool]:
    """Пул к Postgres с пустой таблицей задач перед каждым тестом"""
    pool = await asyncpg.create_pool(postgres_url, min_size=1, max_size=10)
    await ensure_schema(pool)
    await pool.execute("TRUNCATE tasks")
    yield pool
    await pool.close()


@pytest.fixture
def model_loader(monkeypatch):
    """Подменить только загрузку модели, сохранив настоящий путь обработки запроса"""
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
    loader = Mock(
        return_value=InferenceService(
            model=model,
            info=ModelInfo(
                name="toxic-spans-detector", version="1", alias="champion", run_id="run-one"
            ),
        )
    )
    monkeypatch.setattr("src.app.load_inference", loader)
    return loader


@pytest.fixture
async def client(model_loader) -> AsyncIterator[AsyncClient]:
    """HTTP клиент приложения, который запускает lifespan с пулом БД"""
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
