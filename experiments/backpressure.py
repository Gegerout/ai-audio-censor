"""Эксперимент с противодавлением, два прогона с лимитом очереди и без него

Скрипт управляет стендом docker compose. В каждом прогоне источник отправляет одни и те же строки
файла, через заданное время к одному воркеру добавляется второй. Раз в секунду снимаются глубина
очереди, число воркеров, память очереди и брокера, скорость публикации и подтверждений, а источник
пишет свою скорость в лог. После прогона из базы берётся время выполнения каждой задачи
"""

import argparse
import asyncio
import csv
import json
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid5

import aio_pika
import asyncpg
import httpx

from src.broker import QUEUE_POLICY_NAME, TASKS_QUEUE, apply_queue_policy
from src.config import Settings, get_settings
from src.source import TASK_NAMESPACE

RESULTS = Path("docs/experiment")
DATASET_FILE = next(iter(sorted(Path("data/raw").glob("*/train.csv"))), None)
MODES = {"limit": "С лимитом очереди", "nolimit": "Без лимита"}
MEMORY_PATTERN = re.compile(r"([\d.]+)\s*([KMG]i?B)")
UNITS = {"KiB": 1 / 1024, "KB": 1 / 1000, "MiB": 1.0, "MB": 1.0, "GiB": 1024.0, "GB": 1000.0}


@dataclass
class Run:
    """Параметры одного прогона"""

    mode: str
    count: int
    second_worker_at: float
    limit: int | None


def compose(*args: str) -> None:
    """Выполнить команду docker compose и остановиться при ошибке"""
    subprocess.run(["docker", "compose", *args], check=True, capture_output=True, text=True)


def scale_workers(count: int) -> None:
    """Оставить нужное число воркеров и дождаться их готовности"""
    compose("up", "-d", "--no-deps", "--no-build", "--wait", "--scale", f"worker={count}", "worker")


def to_mib(value: str) -> float:
    """Перевести размер из вывода docker stats в мебибайты"""
    match = MEMORY_PATTERN.match(value.strip())
    return float(match.group(1)) * UNITS[match.group(2)] if match else 0.0


def sample_containers(started: float, stop: threading.Event, rows: list[dict]) -> None:
    """Раз в две секунды записывать память контейнеров источника, брокера и воркеров"""
    while not stop.is_set():
        output = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}};{{.MemUsage}}"],
            capture_output=True,
            text=True,
        ).stdout
        moment = round(time.time() - started, 1)
        for line in output.splitlines():
            name, usage = line.split(";", 1)
            for role in ("source", "rabbitmq", "worker"):
                if role in name:
                    rows.append({"t": moment, "role": role, "name": name, "mib": to_mib(usage)})
        stop.wait(2)


async def sample_queue(
    client: httpx.AsyncClient,
    channel: aio_pika.abc.AbstractChannel,
    settings: Settings,
    started: float,
    rows: list[dict],
) -> None:
    """Раз в секунду записывать состояние очереди задач и память брокера

    Глубину очереди и число воркеров брокер отдаёт по AMQP точно и сразу, а счётчики
    Management API обновляются ступеньками, поэтому из него берутся только память и скорости
    """
    base = f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}/api"
    auth = (settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value())
    while True:
        declared = await channel.declare_queue(TASKS_QUEUE.name, passive=True)
        queue = (await client.get(f"{base}/queues/%2F/{TASKS_QUEUE.name}", auth=auth)).json()
        node = (await client.get(f"{base}/nodes?columns=mem_used", auth=auth)).json()[0]
        stats = queue.get("message_stats") or {}
        rows.append(
            {
                "t": round(time.time() - started, 1),
                "ready": declared.declaration_result.message_count,
                "unacked": queue.get("messages_unacknowledged", 0),
                "consumers": declared.declaration_result.consumer_count,
                "queue_memory_mib": queue.get("memory", 0) / 2**20,
                "message_mib": queue.get("message_bytes", 0) / 2**20,
                "broker_memory_mib": node["mem_used"] / 2**20,
                "publish_rate": (stats.get("publish_details") or {}).get("rate", 0.0),
                "ack_rate": (stats.get("ack_details") or {}).get("rate", 0.0),
            }
        )
        await asyncio.sleep(1)


async def set_limit(client: httpx.AsyncClient, settings: Settings, limit: int | None) -> None:
    """Задать лимит очереди политикой или убрать политику совсем"""
    if limit is not None:
        await apply_queue_policy(settings.model_copy(update={"tasks_queue_limit": limit}), client)
        return
    response = await client.delete(
        f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}"
        f"/api/policies/%2F/{QUEUE_POLICY_NAME}",
        auth=(settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value()),
    )
    if response.status_code not in (204, 404):
        response.raise_for_status()


async def purge_queue(client: httpx.AsyncClient, settings: Settings) -> None:
    """Очистить очередь задач перед прогоном"""
    response = await client.delete(
        f"http://{settings.rabbitmq_host}:{settings.rabbitmq_management_port}"
        f"/api/queues/%2F/{TASKS_QUEUE.name}/contents",
        auth=(settings.rabbitmq_user, settings.rabbitmq_password.get_secret_value()),
    )
    response.raise_for_status()


async def run_source(file: Path, name: str, count: int, started: float, rows: list[dict]) -> None:
    """Запустить источник в контейнере и собрать его скорость из логов"""
    process = await asyncio.create_subprocess_exec(
        *("docker", "compose", "run", "--rm", "--no-deps", "source"),
        *("--file", str(file), "--run", name, "--count", str(count)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for raw in process.stdout:
        line = raw.decode(errors="replace")
        if '"source ' not in line:
            continue
        record = json.loads(line[line.index("{") :])
        rows.append(
            {
                "t": round(time.time() - started, 1),
                "msg": record["msg"],
                "published": record.get("published"),
                "rate": record.get("rate"),
                "rejected_recently": record.get("rejected_recently"),
                "rejected": record.get("rejected"),
                "delay_ms": record.get("delay_ms"),
            }
        )
    if await process.wait():
        raise RuntimeError("Источник завершился с ошибкой")


async def connect_db(settings: Settings) -> asyncpg.Connection:
    """Подключиться к базе приложения"""
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        database=settings.postgres_db,
    )


async def task_rows(settings: Settings, file: Path, name: str, count: int, started: float) -> list:
    """Время приёма, начала и конца каждой задачи прогона в секундах от его начала"""
    ids = [uuid5(TASK_NAMESPACE, f"{name}/{file.name}/{row}") for row in range(count)]
    connection = await connect_db(settings)
    try:
        records = await connection.fetch(
            "SELECT status, attempts, enqueued_at, started_at, finished_at FROM tasks"
            " WHERE id = ANY($1::uuid[]) ORDER BY finished_at",
            ids,
        )
    finally:
        await connection.close()
    return [
        {
            "status": record["status"],
            "attempts": record["attempts"],
            "enqueued": _seconds(record["enqueued_at"], started),
            "started": _seconds(record["started_at"], started),
            "finished": _seconds(record["finished_at"], started),
            "latency": (record["finished_at"] - record["enqueued_at"]).total_seconds(),
        }
        for record in records
    ]


def _seconds(moment: datetime | None, started: float) -> float | None:
    """Время события в секундах от начала прогона"""
    return None if moment is None else round(moment.timestamp() - started, 3)


async def wait_done(settings: Settings, file: Path, name: str, count: int) -> None:
    """Дождаться, пока все задачи прогона выполнены"""
    ids = [uuid5(TASK_NAMESPACE, f"{name}/{file.name}/{row}") for row in range(count)]
    connection = await connect_db(settings)
    try:
        query = "SELECT count(*) FROM tasks WHERE status = 'done' AND id = ANY($1::uuid[])"
        # Задачи разбирает воркер в другом процессе, события нет, поэтому база опрашивается
        for _ in range(3600):
            if await connection.fetchval(query, ids) >= count:
                return
            await asyncio.sleep(1)
        raise TimeoutError("Задачи прогона не выполнены за час")
    finally:
        await connection.close()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Сохранить строки в CSV"""
    with path.open("w", newline="", encoding="utf-8") as file:
        # Окончания строк как в остальном репозитории, иначе хуки pre-commit их правят
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


async def run_once(run: Run, settings: Settings, file: Path) -> None:
    """Выполнить один прогон и сохранить его данные"""
    name = f"experiment-{run.mode}-{int(time.time())}"
    out = RESULTS / run.mode
    out.mkdir(parents=True, exist_ok=True)
    scale_workers(1)
    queue_rows, source_rows, container_rows = [], [], []
    connection = await aio_pika.connect(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        login=settings.rabbitmq_user,
        password=settings.rabbitmq_password.get_secret_value(),
    )
    channel = await connection.channel()
    async with httpx.AsyncClient(timeout=5) as client:
        await set_limit(client, settings, run.limit)
        await purge_queue(client, settings)
        started = time.time()
        stop = threading.Event()
        containers = threading.Thread(
            target=sample_containers, args=(started, stop, container_rows), daemon=True
        )
        containers.start()
        sampler = asyncio.create_task(sample_queue(client, channel, settings, started, queue_rows))
        source = asyncio.create_task(run_source(file, name, run.count, started, source_rows))
        await asyncio.sleep(run.second_worker_at)
        second_worker = round(time.time() - started, 1)
        await asyncio.to_thread(scale_workers, 2)
        await source
        await wait_done(settings, file, name, run.count)
        await asyncio.sleep(2)
        sampler.cancel()
        stop.set()
        containers.join()
        await set_limit(client, settings, settings.tasks_queue_limit)
    await connection.close()
    scale_workers(1)
    write_csv(out / "queue.csv", queue_rows)
    write_csv(out / "source.csv", source_rows)
    write_csv(out / "containers.csv", container_rows)
    write_csv(out / "tasks.csv", await task_rows(settings, file, name, run.count, started))
    meta = {**run.__dict__, "run": name, "second_worker_requested_at": second_worker}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки"""
    parser = argparse.ArgumentParser(description="Эксперимент с лимитом очереди и без него")
    parser.add_argument("--count", type=int, default=4000, help="сколько строк отправить")
    parser.add_argument(
        "--second-worker-at", type=float, default=60, help="через сколько секунд добавить воркер"
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    """Выполнить прогоны по очереди"""
    args = parse_args(argv)
    settings = get_settings()
    if DATASET_FILE is None:
        raise SystemExit("Нет train.csv, сначала выполните make eda")
    for mode in args.modes:
        limit = settings.tasks_queue_limit if mode == "limit" else None
        print(f"{MODES[mode]}, {args.count} задач", flush=True)
        await run_once(Run(mode, args.count, args.second_worker_at, limit), settings, DATASET_FILE)


if __name__ == "__main__":
    asyncio.run(main())
