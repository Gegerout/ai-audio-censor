"""Топология RabbitMQ конвейера задач и политика, которая ограничивает очередь задач

Все компоненты берут обменники и очереди отсюда, поэтому объявляют их с одинаковыми параметрами
"""

import logging
import re
import time
from collections.abc import Iterator

import httpx
from faststream.rabbit import (
    Channel,
    ExchangeType,
    QueueType,
    RabbitBroker,
    RabbitExchange,
    RabbitQueue,
)
from faststream.security import SASLPlaintext
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace import SpanKind

from src import tracing
from src.config import Settings
from src.schemas import TaskMessage

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


async def publish_message(
    broker: RabbitBroker, message: TaskMessage, batch: str | None = None
) -> None:
    """Передать контекст публикации вместе с задачей и дождаться подтверждения брокера"""
    with tracing.span(
        "publish",
        kind=SpanKind.PRODUCER,
        attributes={
            "messaging.system": "rabbitmq",
            "messaging.destination.name": TASKS_ROUTING_KEY,
            "task.id": str(message.task_id),
        },
    ):
        headers = {}
        tracing.propagator.inject(headers)
        headers["published_at_ns"] = time.time_ns()
        if batch:
            headers["batch"] = batch
        await broker.publish(
            message,
            exchange=TASKS_EXCHANGE,
            routing_key=TASKS_ROUTING_KEY,
            persist=True,
            correlation_id=str(message.task_id),
            headers=headers,
        )


class BrokerLogger(logging.LoggerAdapter):
    """Добавить correlation_id в записи FastStream о получении и подтверждении сообщения"""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        message = self.extra["current_message"]()
        if message is not None:
            fields = kwargs.get("extra") or {}
            kwargs["extra"] = {**fields, "task_id": message.correlation_id}
            if kwargs.get("exc_info"):
                msg = (
                    "unhandled task error, rejecting to dead letter queue"
                    if "queue" in fields
                    else "broker acknowledgement failed"
                )
        return msg, kwargs


def make_broker(settings: Settings, graceful_timeout: float | None = None) -> RabbitBroker:
    """Создать брокер, логин и пароль передаются отдельно, поэтому спецсимволы не ломают адрес

    graceful_timeout задаёт, сколько при остановке ждать обработчики, которые уже работают
    """
    broker = RabbitBroker(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        security=SASLPlaintext(
            username=settings.rabbitmq_user,
            password=settings.rabbitmq_password.get_secret_value(),
        ),
        graceful_timeout=graceful_timeout,
        # Публикация ждёт подтверждения брокера, а сообщение, которое не попало ни в одну очередь,
        # вызывает ошибку, а не теряется молча
        default_channel=Channel(publisher_confirms=True, on_return_raises=True),
        # Логи FastStream идут в общий JSON формат, а не в его собственный цветной вывод
        logger=BrokerLogger(
            logging.getLogger(__name__),
            {"current_message": lambda: broker.context.get_local("message")},
        ),
        # Запись о каждом сообщении видна только на уровне DEBUG, итог задачи пишет сам компонент
        log_level=logging.DEBUG,
    )
    return broker


def queue_policy(settings: Settings) -> dict:
    """Вернуть политику очереди задач с лимитом длины и отказом в публикации при заполнении"""
    return {
        "pattern": f"^{re.escape(TASKS_QUEUE.name)}$",
        "apply-to": "queues",
        "priority": 0,
        "definition": {"max-length": settings.tasks_queue_limit, "overflow": "reject-publish"},
    }


def observe_queue_limit(settings: Settings, options: CallbackOptions) -> Iterator[Observation]:
    """Прочитать действующее ограничение длины очереди в потоке сбора метрик"""
    try:
        response = httpx.get(
            f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}"
            f"/api/queues/%2F/{TASKS_QUEUE.name}",
            params={"columns": "arguments,effective_policy_definition", "disable_stats": "true"},
            auth=(settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value()),
            timeout=min(1.0, options.timeout_millis / 1000),
        )
        response.raise_for_status()
        queue = response.json()
    except (httpx.HTTPError, ValueError):
        return
    limits = (
        queue["arguments"].get("x-max-length"),
        queue["effective_policy_definition"].get("max-length"),
    )
    yield Observation(min((n for n in limits if n is not None), default=-1))


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


async def declare_dead_letters(broker: RabbitBroker) -> None:
    """Объявить обменник и очередь недоставленных сообщений, повтор безопасен"""
    dead_letter_exchange = await broker.declare_exchange(DEAD_LETTER_EXCHANGE)
    dead_letters = await broker.declare_queue(DEAD_LETTER_QUEUE)
    await dead_letters.bind(dead_letter_exchange)


async def declare_topology(broker: RabbitBroker) -> None:
    """Объявить обменники и очереди задач и недоставленных сообщений, повтор безопасен"""
    await declare_dead_letters(broker)
    tasks_exchange = await broker.declare_exchange(TASKS_EXCHANGE)
    tasks = await broker.declare_queue(TASKS_QUEUE)
    await tasks.bind(tasks_exchange, routing_key=TASKS_ROUTING_KEY)
