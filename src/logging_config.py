"""Структурированные логи в stdout, одна строка на одну запись"""

import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import UUID

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)

# Стандартные поля LogRecord, всё остальное пришло через extra и попадёт в JSON
_RECORD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "color_message"}
_DSN_PASSWORD = re.compile(r'(://[^/@\s:"]*:)[^/\s"]+@')


@contextmanager
def task_context(task_id: UUID | str | None) -> Iterator[None]:
    """Привязать записи к задаче до выхода из обработчика"""
    token = task_id_var.set(str(task_id) if task_id else None)
    try:
        yield
    finally:
        task_id_var.reset(token)


class JsonFormatter(logging.Formatter):
    """Форматирует запись лога в JSON"""

    def __init__(self, service: str = "pipeline") -> None:
        super().__init__()
        self.service = os.getenv("OTEL_SERVICE_NAME") or service
        self.environment = os.getenv("DEPLOYMENT_ENVIRONMENT") or "local"

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "env": self.environment,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RECORD_ATTRS})
        if task_id := task_id_var.get():
            payload["task_id"] = task_id
        if record.exc_info:
            if "task_id" in payload:
                # В тексте исключения модели или базы может быть строка датасета
                kind, _, tb = record.exc_info
                payload["exc_info"] = (
                    "Traceback (most recent call last):\n"
                    + "".join(traceback.format_tb(tb))
                    + kind.__name__
                )
            else:
                payload["exc_info"] = self.formatException(record.exc_info)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        return _DSN_PASSWORD.sub(r"\1***@", line)


def setup_logging(level: str, service: str = "pipeline") -> None:
    """Настроить stdout для приложения и библиотек"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    handler.setLevel(level.upper())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    logging.captureWarnings(True)
    for name in ("uvicorn", "uvicorn.error", "mlflow", "sqlalchemy.engine", "alembic", "huey"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
    # AMQP кадры на DEBUG содержат тело задачи и данные авторизации
    for name in ("aiormq", "aio_pika", "sqlalchemy.engine"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Access лог uvicorn дублирует наш middleware, который пишет ещё и request_id
    logging.getLogger("uvicorn.access").disabled = True
