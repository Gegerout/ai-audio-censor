"""Тесты воркера на TestRabbitBroker и настоящем Postgres"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import Mock
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pandas as pd
import pytest
from faststream import TestApp
from faststream.asgi import AsgiFastStream
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from faststream.rabbit.message import RabbitMessage
from httpx import ASGITransport, AsyncClient

from src import worker
from src.broker import DEAD_LETTER_EXCHANGE, DEAD_LETTER_QUEUE, TASKS_EXCHANGE, TASKS_ROUTING_KEY
from src.config import Settings, get_settings
from src.schemas import TaskMessage, TaskStatus
from src.storage import get_task, start_attempt
from src.worker import create_pool, create_worker, run_inference

ENQUEUED = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
ACK = ("ack", None)
REQUEUE = ("reject", True)
DEAD_LETTER = ("reject", False)


@pytest.fixture
def outcomes(monkeypatch) -> list[tuple[str, bool | None]]:
    """Записать, как воркер завершил каждое сообщение, подтвердил или отклонил"""
    calls = []

    async def ack(self, multiple=False):
        calls.append(ACK)

    async def reject(self, requeue=False):
        calls.append(("reject", requeue))

    async def nack(self, multiple=False, requeue=True):
        calls.append(("nack", requeue))

    monkeypatch.setattr(RabbitMessage, "ack", ack)
    monkeypatch.setattr(RabbitMessage, "reject", reject)
    monkeypatch.setattr(RabbitMessage, "nack", nack)
    return calls


@pytest.fixture
def make_worker(postgres_url, fake_inference, task_pool) -> Callable[..., AsgiFastStream]:
    """Собрать воркер с моделью без MLflow и пулом к тестовому Postgres"""

    def build(loader: Mock | None = None, **overrides) -> AsgiFastStream:
        settings = get_settings().model_copy(
            update={"inference_seconds": 0, "task_retry_delay_seconds": 0, **overrides}
        )
        return create_worker(
            settings,
            load_model=loader or Mock(return_value=fake_inference),
            connect=lambda _: asyncpg.create_pool(postgres_url, min_size=1, max_size=4),
        )

    return build


@asynccontextmanager
async def running(app: AsgiFastStream) -> AsyncIterator[RabbitBroker]:
    """Запустить воркер с хуками старта и остановки на брокере в памяти"""
    async with TestRabbitBroker(app.broker, connect_only=True) as broker, TestApp(app):
        yield broker


async def publish(broker: RabbitBroker, message: TaskMessage | bytes) -> None:
    """Опубликовать сообщение в очередь задач так же, как это делает источник"""
    await broker.publish(message, exchange=TASKS_EXCHANGE, routing_key=TASKS_ROUTING_KEY)


def task(text: str = "You are an idiot.") -> TaskMessage:
    """Создать задачу с новым id"""
    return TaskMessage(task_id=uuid4(), text=text, enqueued_at=ENQUEUED)


def failing_predict(frame):
    """Имитировать сбой модели"""
    raise RuntimeError("model crashed")


async def test_task_result_is_stored_before_ack(make_worker, task_pool, outcomes) -> None:
    """Результат записан в базу и только после этого сообщение подтверждено"""
    message = task()
    stored = []

    async def ack(self, multiple=False):
        stored.append(await get_task(task_pool, message.task_id))
        outcomes.append(ACK)

    app = make_worker()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RabbitMessage, "ack", ack)
        async with running(app) as broker:
            await publish(broker, message)

    assert outcomes == [ACK]
    assert stored[0].status is TaskStatus.DONE
    record = await get_task(task_pool, message.task_id)
    assert record.result == [{"start": 11, "end": 16, "text": "idiot"}]
    assert (record.model_name, record.model_version) == ("toxic-spans-detector", "1")
    assert record.attempts == 1
    assert record.finished_at is not None


async def test_model_is_loaded_once_at_start(make_worker, fake_inference, outcomes) -> None:
    """Модель загружается при старте один раз, а не на каждую задачу"""
    loader = Mock(return_value=fake_inference)
    app = make_worker(loader)

    async with running(app) as broker:
        assert loader.call_count == 1
        for _ in range(3):
            await publish(broker, task())

    assert loader.call_count == 1
    assert fake_inference.model.predict.call_count == 3
    assert outcomes == [ACK] * 3


async def test_worker_declares_dead_letter_topology(make_worker) -> None:
    """Воркер сам объявляет DLQ и её обменник, отклонённые задачи не пропадут"""
    async with running(make_worker()) as broker:
        declarer = broker.config.declarer
        queues = [call.args[0] for call in declarer.declare_queue.await_args_list]
        exchanges = [call.args[0] for call in declarer.declare_exchange.await_args_list]

    assert DEAD_LETTER_QUEUE in queues
    assert DEAD_LETTER_EXCHANGE in exchanges


async def test_model_loading_failure_stops_start(make_worker) -> None:
    """Без модели воркер не стартует и не начинает брать задачи"""
    app = make_worker(Mock(side_effect=ConnectionError("MLflow unavailable")))

    with pytest.raises(ConnectionError):
        async with running(app):
            pass


async def test_duplicate_delivery_skips_inference(make_worker, fake_inference, outcomes) -> None:
    """Повторная доставка выполненной задачи подтверждается без второго inference"""
    message = task()

    async with running(make_worker()) as broker:
        await publish(broker, message)
        await publish(broker, message)

    assert fake_inference.model.predict.call_count == 1
    assert outcomes == [ACK, ACK]


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"task_id": "1", "text": "hello", "enqueued_at": "2026-10-22T12:00:00Z"}',
        b'{"task_id": "00000000-0000-0000-0000-000000000001", "text": "   ",'
        b' "enqueued_at": "2026-10-22T12:00:00Z"}',
        b'{"task_id": "00000000-0000-0000-0000-000000000001", "text": "hello",'
        b' "enqueued_at": "2026-10-22T12:00:00"}',
    ],
)
async def test_malformed_message_goes_to_dead_letters(
    make_worker, fake_inference, task_pool, outcomes, body
) -> None:
    """Некорректное сообщение сразу уходит в DLQ, повтор его не исправит"""
    async with running(make_worker()) as broker:
        await publish(broker, body)

    assert outcomes == [DEAD_LETTER]
    fake_inference.model.predict.assert_not_called()
    assert await task_pool.fetchval("SELECT count(*) FROM tasks") == 0


async def test_model_error_is_retried_then_dead_lettered(
    make_worker, fake_inference, task_pool, outcomes
) -> None:
    """Ошибка модели повторяется до лимита доставок, затем задача проваливается и уходит в DLQ"""
    fake_inference.model.predict.side_effect = failing_predict
    message = task()

    async with running(make_worker(task_max_deliveries=3)) as broker:
        for _ in range(3):
            await publish(broker, message)

    assert outcomes == [REQUEUE, REQUEUE, DEAD_LETTER]
    record = await get_task(task_pool, message.task_id)
    assert record.status is TaskStatus.FAILED
    assert record.attempts == 3
    assert record.error == "ошибка модели RuntimeError"


async def test_retry_after_model_error_succeeds(
    make_worker, fake_inference, task_pool, outcomes
) -> None:
    """Временный сбой модели не теряет задачу, следующая доставка её выполняет"""
    message = task()
    predict = fake_inference.model.predict.side_effect

    async with running(make_worker()) as broker:
        fake_inference.model.predict.side_effect = failing_predict
        await publish(broker, message)
        fake_inference.model.predict.side_effect = predict
        await publish(broker, message)

    assert outcomes == [REQUEUE, ACK]
    record = await get_task(task_pool, message.task_id)
    assert record.status is TaskStatus.DONE
    assert record.attempts == 2


async def test_retry_waits_configured_delay(make_worker, fake_inference, outcomes) -> None:
    """Перед возвратом в очередь воркер ждёт паузу, а не крутит повторы вхолостую"""
    fake_inference.model.predict.side_effect = failing_predict

    async with running(make_worker(task_retry_delay_seconds=0.3)) as broker:
        started = time.perf_counter()
        await publish(broker, task())
        elapsed = time.perf_counter() - started

    assert outcomes == [REQUEUE]
    assert elapsed >= 0.3


async def test_crashed_deliveries_are_counted(
    make_worker, fake_inference, task_pool, outcomes
) -> None:
    """Доставки, на которых воркер упал, тоже учитываются, и задача не крутится бесконечно"""
    message = task()
    for _ in range(3):
        await start_attempt(task_pool, message.task_id, message.text, message.enqueued_at)

    async with running(make_worker(task_max_deliveries=3)) as broker:
        await publish(broker, message)

    assert outcomes == [DEAD_LETTER]
    fake_inference.model.predict.assert_not_called()
    record = await get_task(task_pool, message.task_id)
    assert record.status is TaskStatus.FAILED
    assert record.attempts == 4
    assert record.error == "превышено число доставок"


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("start_attempt", ConnectionRefusedError("db is down")),
        ("complete_task", asyncpg.InterfaceError("connection is closed")),
        ("complete_task", TimeoutError()),
    ],
)
async def test_storage_failure_requeues_task(
    make_worker, task_pool, outcomes, target, error
) -> None:
    """Без записи в базу сообщение не подтверждается, а возвращается в очередь"""
    message = task()

    async def broken(*args, **kwargs):
        raise error

    async with running(make_worker()) as broker:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(worker, target, broken)
            await publish(broker, message)
        await publish(broker, message)

    assert outcomes == [REQUEUE, ACK]
    assert (await get_task(task_pool, message.task_id)).status is TaskStatus.DONE


async def test_unexpected_error_goes_to_dead_letters(make_worker, outcomes, monkeypatch) -> None:
    """Неожиданная ошибка не уходит в бесконечный повтор, политика отправляет её в DLQ"""

    async def broken(*args, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(worker, "complete_task", broken)
    async with running(make_worker()) as broker:
        # Брокер в памяти пробрасывает ошибку публикующему, настоящий только пишет её в лог
        with pytest.raises(KeyError):
            await publish(broker, task())

    assert outcomes == [DEAD_LETTER]


def test_inference_lasts_configured_time(fake_inference) -> None:
    """Inference длится столько, сколько задано в настройках"""
    started = time.perf_counter()
    response = run_inference(fake_inference, "You are an idiot.", 0.3)
    elapsed = time.perf_counter() - started

    assert 0.3 <= elapsed < 0.4
    assert [span.text for span in response.spans] == ["idiot"]


def test_slow_model_is_not_delayed_further(fake_inference) -> None:
    """Если модель сама работает дольше заданного, лишняя пауза не добавляется"""

    def slow_predict(frame):
        time.sleep(0.2)
        return pd.DataFrame({"positions": [[]]})

    fake_inference.model.predict.side_effect = slow_predict
    started = time.perf_counter()
    run_inference(fake_inference, "hello", 0.1)

    assert time.perf_counter() - started < 0.3


def test_prefetch_equals_pool_size_by_default(make_worker) -> None:
    """Брокер отдаёт воркеру столько сообщений, сколько потоков в пуле"""
    app = make_worker(worker_concurrency=3)

    assert app.broker.subscribers[0].channel.prefetch_count == 3


def test_prefetch_can_be_set_explicitly(make_worker) -> None:
    """prefetch_count можно задать отдельно от пула для экспериментов"""
    app = make_worker(worker_concurrency=3, worker_prefetch=10)

    assert app.broker.subscribers[0].channel.prefetch_count == 10


async def test_pool_limits_parallel_inference(make_worker, fake_inference, outcomes) -> None:
    """Одновременно выполняется не больше задач, чем потоков в пуле"""
    active, peak, lock = 0, 0, threading.Lock()

    def predict(frame):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return pd.DataFrame({"positions": [[]]})

    fake_inference.model.predict.side_effect = predict

    async with running(make_worker(worker_concurrency=2)) as broker:
        await asyncio.gather(*(publish(broker, task()) for _ in range(6)))

    assert peak == 2
    assert outcomes == [ACK] * 6


async def test_healthz_answers_during_inference(make_worker, fake_inference, outcomes) -> None:
    """Пока inference идёт в потоке пула, цикл событий свободен и отвечает на проверку живости"""
    entered, release = threading.Event(), threading.Event()
    threads = []

    def predict(frame):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(timeout=3)
        return pd.DataFrame({"positions": [[]]})

    fake_inference.model.predict.side_effect = predict
    app = make_worker()

    async with running(app) as broker:
        pending = asyncio.create_task(publish(broker, task()))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://worker") as http:
                response = await asyncio.wait_for(http.get("/healthz"), timeout=1)
            assert response.status_code == 204
            assert not pending.done()
            assert threads[0] != threading.get_ident()
        finally:
            release.set()
            await pending

    assert outcomes == [ACK]


async def test_stop_waits_for_running_tasks(make_worker, fake_inference, task_pool, outcomes):
    """При остановке воркер дожидается начатой задачи, записывает её и только потом выходит"""
    entered, release = threading.Event(), threading.Event()

    def predict(frame):
        entered.set()
        release.wait(timeout=3)
        return pd.DataFrame({"positions": [[]]})

    fake_inference.model.predict.side_effect = predict
    app = make_worker(worker_graceful_timeout=5)
    message = task()

    async with TestRabbitBroker(app.broker, connect_only=True) as broker:
        await app.start()
        pending = asyncio.create_task(publish(broker, message))
        assert await asyncio.to_thread(entered.wait, 2)
        stopping = asyncio.create_task(app.stop())
        await asyncio.sleep(0.2)
        assert not stopping.done()
        release.set()
        await stopping
        await pending

    assert outcomes == [ACK]
    assert (await get_task(task_pool, message.task_id)).status is TaskStatus.DONE


async def test_create_pool_connects_with_settings(postgres_url) -> None:
    """Пул воркера подключается по настройкам и ограничен размером пула потоков"""
    url = urlsplit(postgres_url)
    settings = Settings(
        postgres_host=url.hostname,
        postgres_port=url.port,
        postgres_user=url.username,
        postgres_password=url.password,
        postgres_db=url.path.lstrip("/"),
        worker_concurrency=3,
    )

    pool = await create_pool(settings)
    try:
        assert await pool.fetchval("SELECT 1") == 1
        assert pool.get_max_size() == 3
    finally:
        await pool.close()
