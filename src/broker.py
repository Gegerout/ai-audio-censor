"""Топология RabbitMQ конвейера задач и политика, которая ограничивает очередь задач

Все компоненты берут обменники и очереди отсюда, поэтому объявляют их с одинаковыми параметрами
"""

import re

import httpx
from faststream.rabbit import (
    ExchangeType,
    QueueType,
    RabbitBroker,
    RabbitExchange,
    RabbitQueue,
)
from faststream.security import SASLPlaintext

from src.config import Settings

TASKS_ROUTING_KEY = "tasks"
TASKS_EXCHANGE = RabbitExchange("tasks", type=ExchangeType.DIRECT, durable=True)
DEAD_LETTER_EXCHANGE = RabbitExchange("tasks.dlx", type=ExchangeType.FANOUT, durable=True)
# Classic, а не quorum, потому что quorum по документации RabbitMQ может превысить лимит длины
# на сообщения в полёте, а глубина очереди задач не должна выходить за лимит
TASKS_QUEUE = RabbitQueue(
    "tasks",
    queue_type=QueueType.CLASSIC,
    durable=True,
    routing_key=TASKS_ROUTING_KEY,
    arguments={"x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name},
)
DEAD_LETTER_QUEUE = RabbitQueue("tasks.dlq", queue_type=QueueType.CLASSIC, durable=True)
QUEUE_POLICY_NAME = "tasks-limit"


def make_broker(settings: Settings) -> RabbitBroker:
    """Создать брокер, логин и пароль передаются отдельно, поэтому спецсимволы не ломают адрес"""
    return RabbitBroker(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        security=SASLPlaintext(
            username=settings.rabbitmq_user,
            password=settings.rabbitmq_password.get_secret_value(),
        ),
    )


def queue_policy(settings: Settings) -> dict:
    """Вернуть политику очереди задач с лимитом длины и отказом в публикации при заполнении"""
    return {
        "pattern": f"^{re.escape(TASKS_QUEUE.name)}$",
        "apply-to": "queues",
        "priority": 0,
        "definition": {"max-length": settings.tasks_queue_limit, "overflow": "reject-publish"},
    }


async def apply_queue_policy(settings: Settings, client: httpx.AsyncClient) -> None:
    """Записать политику в RabbitMQ, повторная запись сразу меняет лимит без потери сообщений"""
    # Лимит задаётся политикой, а не аргументом очереди, иначе для его смены пришлось бы
    # удалить очередь вместе с сообщениями
    response = await client.put(
        f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}"
        f"/api/policies/%2F/{QUEUE_POLICY_NAME}",
        json=queue_policy(settings),
        auth=(settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value()),
    )
    response.raise_for_status()


async def declare_topology(broker: RabbitBroker) -> None:
    """Объявить обменники и очереди задач и недоставленных сообщений, повтор безопасен"""
    tasks_exchange = await broker.declare_exchange(TASKS_EXCHANGE)
    dead_letter_exchange = await broker.declare_exchange(DEAD_LETTER_EXCHANGE)
    tasks = await broker.declare_queue(TASKS_QUEUE)
    await tasks.bind(tasks_exchange, routing_key=TASKS_ROUTING_KEY)
    dead_letters = await broker.declare_queue(DEAD_LETTER_QUEUE)
    await dead_letters.bind(dead_letter_exchange)
