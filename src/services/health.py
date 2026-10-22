"""Сквозная проверка зависимостей с версией и временем ответа каждой

Новая зависимость, например Redis или S3, добавляется ещё одной функцией проверки
и строкой в словаре build_report, контракт ответа не меняется
"""

import asyncio
import logging
import time

import asyncpg
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.schemas import ComponentHealth, HealthResponse, HealthStatus

log = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    """Вернуть время с момента start в миллисекундах"""
    return round((time.perf_counter() - start) * 1000, 2)


async def check_postgres(pool: asyncpg.Pool, settings: Settings) -> ComponentHealth:
    """Запросить версию Postgres и замерить время ответа

    Если БД недоступна, эндпоинт не падает, а возвращает статус fail с именем исключения
    Таймаут обязателен, иначе asyncpg ждёт подключения до 60 секунд
    """
    start = time.perf_counter()
    try:
        async with asyncio.timeout(settings.health_timeout):
            version = await pool.fetchval("SHOW server_version")
    except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        log.warning("postgres health check failed", extra={"error": repr(exc)})
        return ComponentHealth(
            status=HealthStatus.FAIL, latency_ms=_elapsed_ms(start), error=type(exc).__name__
        )
    return ComponentHealth(status=HealthStatus.OK, version=version, latency_ms=_elapsed_ms(start))


async def check_rabbitmq(broker: RabbitBroker, settings: Settings) -> ComponentHealth:
    """Проверить связь с брокером и замерить время ответа, ждать не дольше таймаута"""
    start = time.perf_counter()
    if await broker.ping(settings.health_timeout):
        return ComponentHealth(status=HealthStatus.OK, latency_ms=_elapsed_ms(start))
    log.warning("rabbitmq health check failed")
    return ComponentHealth(
        status=HealthStatus.FAIL, latency_ms=_elapsed_ms(start), error="ConnectionError"
    )


async def build_report(
    pool: asyncpg.Pool, broker: RabbitBroker, settings: Settings
) -> HealthResponse:
    """Собрать отчёт, общий статус ok только если здоровы все зависимости"""
    components = {
        "postgres": await check_postgres(pool, settings),
        "rabbitmq": await check_rabbitmq(broker, settings),
    }
    healthy = all(component.status is HealthStatus.OK for component in components.values())
    return HealthResponse(
        status=HealthStatus.OK if healthy else HealthStatus.FAIL, components=components
    )
