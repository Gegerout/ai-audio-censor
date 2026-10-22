"""Точка сборки приложения FastAPI"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request, Response
from starlette.concurrency import run_in_threadpool

from src.api import healthz, process, v1
from src.config import get_app_version, get_settings
from src.logging_config import request_id_var, setup_logging
from src.services.inference import load_inference

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить модель и создать пул БД до начала обработки запросов"""
    settings = get_settings()
    app.state.inference = await run_in_threadpool(load_inference, settings)
    info = app.state.inference.info
    log.info(
        "model loaded",
        extra={
            "model_name": info.name,
            "model_version": info.version,
            "model_alias": info.alias,
            "training_run_id": info.run_id,
        },
    )
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
    log.info("application started", extra={"version": app.version})
    try:
        yield
    finally:
        await pool.close()
        app.state.inference = None
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


def create_app() -> FastAPI:
    """Собрать приложение, подключить логи, роуты и middleware"""
    setup_logging(get_settings().log_level)
    app = FastAPI(title="AI Audio Censor", version=get_app_version(), lifespan=lifespan)
    app.middleware("http")(log_requests)
    app.include_router(healthz.router)
    app.include_router(v1.router)
    app.include_router(process.router)
    return app


app = create_app()
