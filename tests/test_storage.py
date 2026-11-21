"""Тесты хранилища задач на настоящем Postgres, включая одновременные доставки"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import pytest

from src.schemas import TaskStatus
from src.storage import (
    complete_task,
    create_pending,
    ensure_schema,
    fail_task,
    get_task,
    start_attempt,
)

ENQUEUED = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
SPANS = [{"start": 11, "end": 16, "text": "idiot"}]


async def test_schema_is_created_once_by_concurrent_processes(postgres_url) -> None:
    """Несколько процессов одновременно создают таблицу без ошибки в системном каталоге"""
    pools = [await asyncpg.create_pool(postgres_url, min_size=1, max_size=1) for _ in range(5)]
    await pools[0].execute("DROP TABLE IF EXISTS tasks")

    await asyncio.gather(*(ensure_schema(pool) for pool in pools))

    assert await pools[0].fetchval("SELECT to_regclass('tasks') IS NOT NULL")
    for pool in pools:
        await pool.close()


async def test_first_delivery_creates_processing_task(task_pool) -> None:
    """Первая доставка создаёт задачу в обработке с одной попыткой"""
    task_id = uuid4()

    attempt = await start_attempt(task_pool, task_id, "You are an idiot.", ENQUEUED)

    assert attempt.status is TaskStatus.PROCESSING
    assert attempt.attempts == 1
    record = await get_task(task_pool, task_id)
    assert record.enqueued_at == ENQUEUED
    assert record.started_at is not None


async def test_each_redelivery_increments_attempts(task_pool) -> None:
    """Каждая повторная доставка учитывается до начала обработки"""
    task_id = uuid4()

    attempts = [
        (await start_attempt(task_pool, task_id, "text", ENQUEUED)).attempts for _ in range(3)
    ]

    assert attempts == [1, 2, 3]


async def test_concurrent_deliveries_keep_one_row(task_pool) -> None:
    """Одновременные доставки одной задачи не создают дубликатов и не теряют попытки"""
    task_id = uuid4()

    results = await asyncio.gather(
        *(start_attempt(task_pool, task_id, "text", ENQUEUED) for _ in range(10))
    )

    assert sorted(result.attempts for result in results) == list(range(1, 11))
    assert await task_pool.fetchval("SELECT count(*) FROM tasks WHERE id = $1", task_id) == 1


async def test_result_is_saved_only_once(task_pool) -> None:
    """Повторное сохранение результата ничего не меняет"""
    task_id = uuid4()
    await start_attempt(task_pool, task_id, "You are an idiot.", ENQUEUED)

    first = await complete_task(task_pool, task_id, SPANS, "toxic-spans-detector", "1")
    second = await complete_task(task_pool, task_id, [], "toxic-spans-detector", "3")

    assert (first, second) == (True, False)
    record = await get_task(task_pool, task_id)
    assert record.status is TaskStatus.DONE
    assert record.result == SPANS
    assert record.model_version == "1"
    assert record.finished_at is not None


async def test_concurrent_completions_store_one_result(task_pool) -> None:
    """Из одновременных сохранений одного результата срабатывает ровно одно"""
    task_id = uuid4()
    await start_attempt(task_pool, task_id, "text", ENQUEUED)

    saved = await asyncio.gather(
        *(complete_task(task_pool, task_id, SPANS, "model", str(n)) for n in range(5))
    )

    assert saved.count(True) == 1


async def test_delivery_after_done_does_not_restart_task(task_pool) -> None:
    """Доставка уже выполненной задачи возвращает готовый статус и не считается попыткой"""
    task_id = uuid4()
    await start_attempt(task_pool, task_id, "text", ENQUEUED)
    await complete_task(task_pool, task_id, SPANS, "model", "1")

    attempt = await start_attempt(task_pool, task_id, "text", ENQUEUED)

    assert attempt.status is TaskStatus.DONE
    assert attempt.attempts == 1


async def test_pending_task_from_api_becomes_processing(task_pool) -> None:
    """Задача, принятая через API, переходит в обработку и сохраняет время приёма"""
    task_id = uuid4()
    async with task_pool.acquire() as connection, connection.transaction():
        await create_pending(connection, task_id, "text", ENQUEUED)

    assert (await get_task(task_pool, task_id)).status is TaskStatus.PENDING
    attempt = await start_attempt(task_pool, task_id, "text", datetime.now(UTC))

    assert attempt == type(attempt)(status=TaskStatus.PROCESSING, attempts=1)
    assert (await get_task(task_pool, task_id)).enqueued_at == ENQUEUED


async def test_rolled_back_pending_task_leaves_nothing(task_pool) -> None:
    """Если транзакцию откатили после отказа брокера, задачи в хранилище нет"""
    task_id = uuid4()

    with pytest.raises(RuntimeError):
        async with task_pool.acquire() as connection, connection.transaction():
            await create_pending(connection, task_id, "text", ENQUEUED)
            raise RuntimeError("брокер отказал в публикации")

    assert await get_task(task_pool, task_id) is None


async def test_failed_task_keeps_error_but_done_task_is_not_failed(task_pool) -> None:
    """Ошибка сохраняется у невыполненной задачи, готовую задачу провалить нельзя"""
    failed_id, done_id = uuid4(), uuid4()
    await start_attempt(task_pool, failed_id, "text", ENQUEUED)
    await start_attempt(task_pool, done_id, "text", ENQUEUED)
    await complete_task(task_pool, done_id, SPANS, "model", "1")

    assert await fail_task(task_pool, failed_id, "превышено число попыток")
    assert not await fail_task(task_pool, done_id, "поздняя ошибка")

    failed = await get_task(task_pool, failed_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.error == "превышено число попыток"
    assert (await get_task(task_pool, done_id)).status is TaskStatus.DONE


async def test_unknown_task_is_not_found(task_pool) -> None:
    """Неизвестная задача возвращается как отсутствующая"""
    assert await get_task(task_pool, uuid4()) is None
