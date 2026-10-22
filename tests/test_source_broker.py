"""Переполнение очереди на настоящем RabbitMQ, источник замедляется и не теряет задачи"""

import asyncio
import time
from collections.abc import AsyncIterator
from uuid import uuid5

import aio_pika
import pytest
from faststream.rabbit import RabbitBroker

from src.broker import TASKS_QUEUE, declare_topology, make_broker
from src.config import Settings, get_settings
from src.schemas import TaskMessage
from src.source import TASK_NAMESPACE, Source, SourceTask

LIMIT = 5
COUNT = 30
# Столько потребитель обрабатывает одно сообщение
CONSUMER_SECONDS = 0.05


@pytest.fixture
def settings(rabbitmq_settings) -> Settings:
    """Брокер из контейнера, маленький лимит очереди и короткие паузы источника"""
    return get_settings().model_copy(
        update={
            **rabbitmq_settings,
            "tasks_queue_limit": LIMIT,
            "source_initial_delay_seconds": 0.01,
            "source_max_delay_seconds": 0.2,
            "source_report_seconds": 60,
        }
    )


@pytest.fixture
async def broker(settings, limited_tasks_queue) -> AsyncIterator[RabbitBroker]:
    """Брокер источника, подключённый к очереди с лимитом"""
    broker = make_broker(settings)
    await broker.connect()
    await declare_topology(broker)
    yield broker
    await broker.stop()


def tasks(count: int) -> list[SourceTask]:
    """Строки файла с id по номеру строки"""
    return [
        SourceTask(uuid5(TASK_NAMESPACE, f"overflow/{row}"), f"text {row}", row)
        for row in range(count)
    ]


async def ready(channel: aio_pika.abc.AbstractChannel) -> int:
    """Сколько задач ждёт в очереди"""
    queue = await channel.declare_queue(TASKS_QUEUE.name, passive=True)
    return queue.declaration_result.message_count


async def test_source_slows_to_consumer_speed_without_losing_tasks(
    settings, broker, rabbit_connection
) -> None:
    """Очередь доходит до лимита и не превышает его, источник ждёт потребителя и ничего не теряет"""
    channel = await rabbit_connection.channel()
    await channel.set_qos(prefetch_count=1)
    received = []

    async def slow_consumer(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        await asyncio.sleep(CONSUMER_SECONDS)
        task = TaskMessage.model_validate_json(message.body)
        received.append((task.task_id, message.delivery_mode))
        await message.ack()

    await (await channel.get_queue(TASKS_QUEUE.name)).consume(slow_consumer)
    probe = await rabbit_connection.channel()
    depths = []

    async def watch() -> None:
        while True:
            depths.append(await ready(probe))
            await asyncio.sleep(0.01)

    watcher = asyncio.create_task(watch())
    sent = tasks(COUNT)
    started = time.perf_counter()
    stats = await Source(broker, settings).run(sent)
    elapsed = time.perf_counter() - started
    for _ in range(100):
        if len(received) == COUNT:
            break
        await asyncio.sleep(0.05)
    watcher.cancel()

    assert stats.published == COUNT
    assert stats.rejected > 0
    assert max(depths) == LIMIT
    # Каждая задача пришла ровно один раз и в том порядке, в каком строки идут в файле
    assert [task_id for task_id, _ in received] == [task.task_id for task in sent]
    assert {mode for _, mode in received} == {aio_pika.DeliveryMode.PERSISTENT}
    # Источник не мог закончить раньше, чем потребитель освободил место под последние задачи
    assert elapsed >= (COUNT - LIMIT - 1) * CONSUMER_SECONDS * 0.9


async def test_source_waits_without_consumer_instead_of_spinning(
    settings, broker, rabbit_connection
) -> None:
    """Без потребителя источник заполняет очередь до лимита и ждёт с растущей паузой"""
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(1.0, stop.set)

    stats = await Source(broker, settings, stop).run(tasks(10))

    assert (stats.published, stats.next_row) == (LIMIT, LIMIT)
    # Холостой цикл дал бы тысячи отказов за секунду, пауза до 0,2 секунды даёт около десяти
    assert 3 <= stats.rejected <= 15
    assert await ready(await rabbit_connection.channel()) == LIMIT
