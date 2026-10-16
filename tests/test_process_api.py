"""Проверки inference через HTTP и жизненного цикла модели"""

import asyncio
import threading
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.services.inference import InferenceService


async def test_process_returns_spans_and_loaded_model(client, model_loader):
    """Несколько запросов используют одну модель и сохраняют позиции в исходном тексте"""
    response = await client.post("/process", json={"text": "я😊  IDIOT\nhello"})
    assert response.status_code == 200
    assert response.json() == {
        "model": {
            "name": "toxic-spans-detector",
            "version": "1",
            "alias": "champion",
            "run_id": "run-one",
        },
        "spans": [{"start": 4, "end": 9, "text": "IDIOT"}],
    }
    assert len(response.headers["X-Request-ID"]) == 32
    plain = await client.post("/process", json={"text": "Have a nice day"})
    assert plain.json()["spans"] == []
    assert model_loader.call_count == 1
    assert model_loader.return_value.model.predict.call_count == 2


@pytest.mark.parametrize(
    "body",
    [{}, {"text": ""}, {"text": " \n\t "}, {"text": None}, {"text": 123}, {"text": "x" * 10001}],
)
async def test_invalid_input_does_not_call_model(client, model_loader, body):
    """Невалидный текст отклоняется до запуска inference"""
    response = await client.post("/process", json=body)
    assert response.status_code == 422
    model_loader.return_value.model.predict.assert_not_called()


async def test_inference_does_not_block_healthz(client, model_loader):
    """Пока предсказание ждёт в рабочем потоке, другой запрос получает ответ"""
    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()
    threads = []

    def predict(frame):
        """Удержать рабочий поток до завершения проверки другого эндпоинта"""
        threads.append(threading.get_ident())
        entered.set()
        release.wait(timeout=3)
        return pd.DataFrame({"positions": [[]]})

    model_loader.return_value.model.predict.side_effect = predict
    pending = asyncio.create_task(client.post("/process", json={"text": "hello"}))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert threads == [threads[0]] and threads[0] != event_loop_thread
        response = await asyncio.wait_for(client.get("/healthz"), timeout=1)
        assert response.status_code == 200
        assert not pending.done()
    finally:
        release.set()
        result = await pending
    assert result.status_code == 200


async def test_restart_loads_new_version(model_loader):
    """Новая версия появляется только после следующего запуска lifespan"""
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/process", json={"text": "hello"})
            previous = model_loader.return_value
            model_loader.return_value = InferenceService(
                model=previous.model,
                info=previous.info.model_copy(update={"version": "3", "run_id": "run-three"}),
            )
            still_first = await client.post("/process", json={"text": "hello"})
            assert first.json()["model"]["version"] == still_first.json()["model"]["version"] == "1"
    assert app.state.inference is None
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            third = await client.post("/process", json={"text": "hello"})
            assert third.json()["model"]["version"] == "3"
    assert model_loader.call_count == 2


async def test_model_loading_failure_stops_startup(model_loader, monkeypatch):
    """Приложение не объявляет готовность и не создаёт пул БД без модели"""
    model_loader.side_effect = ConnectionError("MLflow unavailable")
    pool = AsyncMock()
    monkeypatch.setattr("src.app.asyncpg.create_pool", pool)
    with pytest.raises(ConnectionError, match="MLflow unavailable"):
        async with LifespanManager(app):
            pytest.fail("Приложение не должно запускаться")
    pool.assert_not_called()
