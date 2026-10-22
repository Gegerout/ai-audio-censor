"""Подготовка RabbitMQ перед запуском конвейера, лимит очереди и объявление очередей"""

import asyncio
import logging

import httpx

from src.broker import apply_queue_policy, declare_topology, make_broker
from src.config import Settings, get_settings
from src.logging_config import setup_logging

log = logging.getLogger(__name__)


async def setup_broker(settings: Settings) -> None:
    """Сначала записать политику, затем объявить очереди, чтобы очередь не жила без лимита"""
    async with httpx.AsyncClient(timeout=10) as client:
        await apply_queue_policy(settings, client)
    broker = make_broker(settings)
    await broker.start()
    try:
        await declare_topology(broker)
    finally:
        await broker.stop()
    log.info("rabbitmq configured", extra={"tasks_queue_limit": settings.tasks_queue_limit})


def main() -> None:
    """Настроить логи и выполнить подготовку брокера"""
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(setup_broker(settings))


if __name__ == "__main__":
    main()
