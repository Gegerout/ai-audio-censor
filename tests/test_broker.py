"""Тесты топологии RabbitMQ и подготовки брокера"""

import base64
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from faststream.rabbit import QueueType, RabbitBroker

from src import broker_setup
from src.broker import (
    DEAD_LETTER_EXCHANGE,
    DEAD_LETTER_QUEUE,
    TASKS_EXCHANGE,
    TASKS_QUEUE,
    apply_queue_policy,
    declare_topology,
    make_broker,
    queue_policy,
)
from src.config import get_settings


@pytest.fixture
def settings():
    """Настройки из окружения, заданного в conftest, с отдельным лимитом очереди"""
    return get_settings().model_copy(update={"tasks_queue_limit": 7})


def test_queue_policy_limits_only_tasks_queue(settings) -> None:
    """Политика ограничивает только очередь задач и отказывает в публикации при заполнении"""
    policy = queue_policy(settings)

    assert policy["pattern"] == "^tasks$"
    assert policy["apply-to"] == "queues"
    assert policy["definition"] == {"max-length": 7, "overflow": "reject-publish"}


def test_tasks_queue_is_durable_classic_with_dead_letters() -> None:
    """Очередь задач переживает перезапуск брокера и отправляет отклонённые сообщения в DLX"""
    assert TASKS_QUEUE.arguments["x-queue-type"] == QueueType.CLASSIC.value
    assert TASKS_QUEUE.durable
    assert TASKS_QUEUE.arguments["x-dead-letter-exchange"] == DEAD_LETTER_EXCHANGE.name
    assert DEAD_LETTER_QUEUE.durable


async def test_apply_queue_policy_puts_policy_with_credentials(settings) -> None:
    """Политика записывается через HTTP API с логином и паролем из настроек"""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Запомнить запрос и ответить как RabbitMQ при успешной записи"""
        requests.append(request)
        return httpx.Response(201)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await apply_queue_policy(settings, client)

    request = requests[0]
    assert request.method == "PUT"
    assert request.url.host == "127.0.0.1"
    assert request.url.port == 15672
    assert request.url.raw_path == b"/api/policies/%2F/tasks-limit"
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"test:test").decode()
    assert json.loads(request.content) == queue_policy(settings)


async def test_apply_queue_policy_fails_on_rejected_request(settings) -> None:
    """Если RabbitMQ не принял политику, подготовка падает, а не запускает конвейер без лимита"""
    transport = httpx.MockTransport(lambda request: httpx.Response(401))

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await apply_queue_policy(settings, client)


async def test_declare_topology_binds_tasks_and_dead_letter_queues() -> None:
    """Очередь задач связана с обменником по ключу, DLQ связана с обменником недоставленных"""
    exchanges = {}
    queues = {}
    broker = Mock()
    broker.declare_exchange = AsyncMock(
        side_effect=lambda exchange: exchanges.setdefault(exchange.name, Mock())
    )
    broker.declare_queue = AsyncMock(
        side_effect=lambda queue: queues.setdefault(queue.name, AsyncMock())
    )

    await declare_topology(broker)

    queues[TASKS_QUEUE.name].bind.assert_awaited_once_with(
        exchanges[TASKS_EXCHANGE.name], routing_key="tasks"
    )
    queues[DEAD_LETTER_QUEUE.name].bind.assert_awaited_once_with(
        exchanges[DEAD_LETTER_EXCHANGE.name]
    )


def test_make_broker_creates_rabbit_broker(settings) -> None:
    """Брокер создаётся из настроек без адреса с паролем внутри"""
    assert isinstance(make_broker(settings), RabbitBroker)


async def test_setup_applies_policy_before_declaring_queues(settings, monkeypatch) -> None:
    """Лимит записывается раньше очередей, поэтому очередь задач не существует без лимита"""
    calls = []
    broker = Mock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(
        broker_setup, "apply_queue_policy", AsyncMock(side_effect=lambda *_: calls.append("policy"))
    )
    monkeypatch.setattr(
        broker_setup, "declare_topology", AsyncMock(side_effect=lambda _: calls.append("queues"))
    )
    monkeypatch.setattr(broker_setup, "make_broker", Mock(return_value=broker))

    await broker_setup.setup_broker(settings)

    assert calls == ["policy", "queues"]
    broker.stop.assert_awaited_once()


async def test_setup_closes_connection_when_declaring_fails(settings, monkeypatch) -> None:
    """Соединение с брокером закрывается, даже если объявить очереди не удалось"""
    broker = Mock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(broker_setup, "apply_queue_policy", AsyncMock())
    monkeypatch.setattr(broker_setup, "declare_topology", AsyncMock(side_effect=RuntimeError))
    monkeypatch.setattr(broker_setup, "make_broker", Mock(return_value=broker))

    with pytest.raises(RuntimeError):
        await broker_setup.setup_broker(settings)

    broker.stop.assert_awaited_once()


def test_main_runs_setup_with_settings(monkeypatch) -> None:
    """Команда настраивает логи и запускает подготовку брокера с настройками окружения"""
    setup = AsyncMock()
    monkeypatch.setattr(broker_setup, "setup_broker", setup)
    monkeypatch.setattr(broker_setup, "setup_logging", Mock())

    broker_setup.main()

    setup.assert_awaited_once_with(get_settings())
