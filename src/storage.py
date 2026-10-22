"""Хранилище задач в Postgres, идемпотентность обеспечивает первичный ключ задачи

Каждая операция выполняется одним атомарным запросом, поэтому повторная доставка сообщения
не создаёт вторую запись и не перезаписывает готовый результат
"""

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import asyncpg

from src.schemas import TaskStatus

# Номер блокировки, под которой процессы по очереди создают таблицу
SCHEMA_LOCK_ID = 7_301_001
STATUSES = ", ".join(f"'{status.value}'" for status in TaskStatus)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks (
    id uuid PRIMARY KEY,
    text text NOT NULL,
    status text NOT NULL CHECK (status IN ({STATUSES})),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    result jsonb,
    model_name text,
    model_version text,
    error text,
    enqueued_at timestamptz NOT NULL,
    started_at timestamptz,
    finished_at timestamptz
)
"""


@dataclass(frozen=True)
class Attempt:
    """Состояние задачи после начала очередной попытки обработки"""

    status: TaskStatus
    attempts: int


@dataclass(frozen=True)
class TaskRecord:
    """Задача из хранилища вместе с результатом и временем этапов"""

    id: UUID
    text: str
    status: TaskStatus
    attempts: int
    result: list[dict] | None
    model_name: str | None
    model_version: str | None
    error: str | None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Создать таблицу задач, если её ещё нет"""
    async with pool.acquire() as connection, connection.transaction():
        # Одновременный CREATE TABLE IF NOT EXISTS из двух процессов может упасть с ошибкой
        # уникальности в системном каталоге, блокировка выстраивает процессы по очереди
        await connection.execute("SELECT pg_advisory_xact_lock($1)", SCHEMA_LOCK_ID)
        await connection.execute(SCHEMA)


async def create_pending(
    connection: asyncpg.Connection, task_id: UUID, text: str, enqueued_at: datetime
) -> None:
    """Записать принятую задачу в открытой транзакции, при отказе брокера её откатывают"""
    await connection.execute(
        "INSERT INTO tasks (id, text, status, attempts, enqueued_at) VALUES ($1, $2, $3, 0, $4)",
        task_id,
        text,
        TaskStatus.PENDING,
        enqueued_at,
    )


async def start_attempt(
    pool: asyncpg.Pool, task_id: UUID, text: str, enqueued_at: datetime
) -> Attempt:
    """Создать задачу или учесть ещё одну доставку, готовую задачу не трогать"""
    row = await pool.fetchrow(
        """
        INSERT INTO tasks (id, text, status, attempts, enqueued_at, started_at)
        VALUES ($1, $2, $3, 1, $4, now())
        ON CONFLICT (id) DO UPDATE SET
            attempts = tasks.attempts + CASE WHEN tasks.status = $5 THEN 0 ELSE 1 END,
            status = CASE WHEN tasks.status = $5 THEN tasks.status ELSE $3 END,
            started_at = COALESCE(tasks.started_at, now())
        RETURNING status, attempts
        """,
        task_id,
        text,
        TaskStatus.PROCESSING,
        enqueued_at,
        TaskStatus.DONE,
    )
    return Attempt(status=TaskStatus(row["status"]), attempts=row["attempts"])


async def complete_task(
    pool: asyncpg.Pool, task_id: UUID, spans: list[dict], model_name: str, model_version: str
) -> bool:
    """Сохранить результат один раз, повторное сохранение ничего не меняет"""
    status = await pool.execute(
        """
        UPDATE tasks SET status = $2, result = $3::jsonb, model_name = $4, model_version = $5,
            error = NULL, finished_at = now()
        WHERE id = $1 AND status <> $2
        """,
        task_id,
        TaskStatus.DONE,
        json.dumps(spans, ensure_ascii=False),
        model_name,
        model_version,
    )
    return status == "UPDATE 1"


async def fail_task(pool: asyncpg.Pool, task_id: UUID, error: str) -> bool:
    """Отметить задачу проваленной, если она ещё не выполнена"""
    status = await pool.execute(
        """
        UPDATE tasks SET status = $2, error = $3, finished_at = now()
        WHERE id = $1 AND status <> $4
        """,
        task_id,
        TaskStatus.FAILED,
        error,
        TaskStatus.DONE,
    )
    return status == "UPDATE 1"


async def get_task(pool: asyncpg.Pool, task_id: UUID) -> TaskRecord | None:
    """Вернуть задачу по идентификатору или None, если её нет"""
    row = await pool.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
    if row is None:
        return None
    values = dict(row)
    values["status"] = TaskStatus(values["status"])
    if values["result"] is not None:
        values["result"] = json.loads(values["result"])
    return TaskRecord(**values)
