"""Точка сборки приложения FastAPI"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request, Response

from src.api import healthz, tasks, v1
from src.broker import declare_topology, make_broker
from src.config import Settings, get_app_version, get_settings
from src.logging_config import request_id_var, setup_logging
from src.services.tasks import STORAGE_ERRORS
from src.storage import ensure_schema

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Создать пул БД и подключиться к брокеру до начала обработки запросов"""
    settings: Settings = app.state.settings
    # Пул с min_size 0 не подключается при старте, поэтому приложение поднимется и без БД,
    # а /api/v1/health покажет, что она недоступна
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        database=settings.postgres_db,
        min_size=0,
    )
    app.state.db_pool = pool
    broker = app.state.broker
    try:
        # Без брокера задачи принимать некуда, поэтому приложение не стартует
        await broker.connect()
        await declare_topology(broker)
        try:
            await ensure_schema(pool)
        except STORAGE_ERRORS as exc:
            # Таблицу создаст воркер или следующий старт, пока база недоступна, API отвечает 503
            log.warning("task table not ensured", extra={"error": repr(exc)})
        log.info("application started", extra={"version": app.version})
        yield
    finally:
        await broker.stop()
        await pool.close()
        log.info("application stopped")


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Выдать запросу request_id и залогировать метод, путь, статус и длительность"""
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        # Healthcheck Docker дёргает /healthz каждые 10 секунд, на INFO он забил бы полезные логи
        level = logging.DEBUG if request.url.path == "/healthz" else logging.INFO
        log.log(
            level,
            "request handled",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
    except Exception:
        # Access лог uvicorn выключен, без этой записи упавший запрос пропадёт из логов
        log.exception(
            "request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        raise
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение, подключить логи, роуты и middleware"""
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="AI Audio Censor", version=get_app_version(), lifespan=lifespan)
    app.state.settings = settings
    app.state.broker = make_broker(settings)
    app.middleware("http")(log_requests)
    app.include_router(healthz.router)
    app.include_router(v1.router)
    app.include_router(tasks.router)
    return app


app = create_app()
