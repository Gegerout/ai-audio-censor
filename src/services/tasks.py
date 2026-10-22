"""Приём задач от клиентов API и оценка времени, через которое стоит повторить запрос

API не ждёт, пока в очереди освободится место
Если брокер отказал, задача не принимается и строка в базе откатывается
Клиент сразу получает 503 и время, через которое стоит повторить запрос
"""

import asyncio
import math
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
from aiormq.exceptions import (
    AMQPConnectionError,
    ChannelInvalidStateError,
    DeliveryError,
    PublishError,
)
from faststream.rabbit import RabbitBroker

from src.broker import TASKS_EXCHANGE, TASKS_QUEUE, TASKS_ROUTING_KEY
from src.config import Settings
from src.schemas import TaskMessage
from src.storage import create_pending

BROKER_ERRORS = (AMQPConnectionError, ChannelInvalidStateError, ConnectionError)
STORAGE_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError)


class TaskRejectedError(Exception):
    """Задачу нельзя принять сейчас, клиенту стоит повторить через retry_after секунд"""

    def __init__(self, reason: str, retry_after: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class BrokerRefusedError(Exception):
    """Брокер не принял публикацию, queue_full отличает заполненную очередь от сбоя"""

    def __init__(self, reason: str, queue_full: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.queue_full = queue_full


def retry_after_seconds(messages: int, consumers: int, settings: Settings) -> int:
    """Оценить, за сколько секунд воркеры разберут задачи, которые уже ждут в очереди

    Задача, принятая раньше, всё равно ждала бы их, поэтому повтор до этого срока
    результат не ускорит, а только добавит нагрузки
    """
    if consumers == 0:
        # Без воркеров очередь не разбирается, оценить время нельзя
        return settings.tasks_retry_after_max_seconds
    if settings.inference_seconds == 0:
        return 1
    rate = consumers * settings.worker_concurrency / settings.inference_seconds
    return min(settings.tasks_retry_after_max_seconds, max(1, math.ceil(messages / rate)))


async def queue_load(broker: RabbitBroker) -> tuple[int, int]:
    """Число задач в очереди и число воркеров прямо сейчас

    Повторное объявление очереди возвращает свежие счётчики от самого брокера по AMQP,
    в отличие от статистики Management API, которая отстаёт на интервал обновления
    """
    queue = await broker.declare_queue(TASKS_QUEUE)
    declared = await queue.declare()
    return declared.message_count, declared.consumer_count


async def estimate_retry_after(broker: RabbitBroker, settings: Settings) -> int:
    """Время повтора для заполненной очереди, при сбое оценки берётся верхняя граница"""
    try:
        async with asyncio.timeout(settings.tasks_timeout_seconds):
            messages, consumers = await queue_load(broker)
    except (*BROKER_ERRORS, TimeoutError):
        return settings.tasks_retry_after_max_seconds
    return retry_after_seconds(messages, consumers, settings)


async def publish_task(broker: RabbitBroker, message: TaskMessage) -> None:
    """Опубликовать задачу и дождаться подтверждения, отказ брокера отделить от сбоя базы"""
    try:
        await broker.publish(
            message, exchange=TASKS_EXCHANGE, routing_key=TASKS_ROUTING_KEY, persist=True
        )
    except PublishError as exc:
        raise BrokerRefusedError("Очередь задач не объявлена") from exc
    except DeliveryError as exc:
        raise BrokerRefusedError("Очередь задач заполнена, повторите позже", True) from exc
    except BROKER_ERRORS as exc:
        raise BrokerRefusedError("Брокер недоступен") from exc


async def submit_task(
    pool: asyncpg.Pool, broker: RabbitBroker, settings: Settings, text: str
) -> UUID:
    """Записать задачу и опубликовать её, при отказе брокера запись откатывается"""
    task_id = uuid4()
    enqueued_at = datetime.now(UTC)
    message = TaskMessage(task_id=task_id, text=text, enqueued_at=enqueued_at)
    cap = settings.tasks_retry_after_max_seconds
    try:
        async with asyncio.timeout(settings.tasks_timeout_seconds):
            async with pool.acquire() as connection, connection.transaction():
                await create_pending(connection, task_id, text, enqueued_at)
                # Транзакция фиксируется только после подтверждения брокера, при отказе
                # исключение откатывает её, и в базе не остаётся задачи в статусе pending
                await publish_task(broker, message)
    except BrokerRefusedError as exc:
        retry_after = await estimate_retry_after(broker, settings) if exc.queue_full else cap
        raise TaskRejectedError(exc.reason, retry_after) from exc
    except TimeoutError as exc:
        raise TaskRejectedError("База или брокер не ответили вовремя", cap) from exc
    except STORAGE_ERRORS as exc:
        raise TaskRejectedError("База недоступна", cap) from exc
    return task_id
