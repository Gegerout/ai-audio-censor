"""Контракты API и сообщений очереди на Pydantic"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, BaseModel, Field


def reject_blank(value: str) -> str:
    """Отклонить пустое содержимое, сохранив исходные пробелы и позиции"""
    if not value.strip():
        raise ValueError("Текст не должен состоять только из пробелов")
    return value


# Один тип текста для API и очереди, поэтому проверки входа везде одинаковые
TaskText = Annotated[str, Field(min_length=1, max_length=10000), AfterValidator(reject_blank)]


class HealthStatus(StrEnum):
    """Статус приложения или отдельной зависимости"""

    OK = "ok"
    FAIL = "fail"


class TaskStatus(StrEnum):
    """Состояние задачи в хранилище"""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class LivenessResponse(BaseModel):
    """Ответ /healthz"""

    status: HealthStatus


class VersionResponse(BaseModel):
    """Ответ /api/v1/version"""

    version: str


class ComponentHealth(BaseModel):
    """Состояние одной зависимости, её версия и время ответа"""

    status: HealthStatus
    version: str | None = None
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    """Ответ /api/v1/health с общим статусом и состоянием каждой зависимости"""

    status: HealthStatus
    components: dict[str, ComponentHealth]


class TaskRequest(BaseModel):
    """Текст, в котором нужно найти токсичные фрагменты"""

    text: TaskText


class TaskMessage(BaseModel):
    """Задача в очереди RabbitMQ, её публикуют источник и API, а выполняет воркер"""

    task_id: UUID
    text: TaskText
    # Время приёма задачи, от него считается полное время выполнения вместе с ожиданием
    enqueued_at: AwareDatetime


class ModelInfo(BaseModel):
    """Версия загруженной модели и alias на момент старта воркера"""

    name: str
    version: str
    alias: str
    run_id: str


class TextSpan(BaseModel):
    """Выделенный фрагмент с включённым началом и исключённым концом"""

    start: int
    end: int
    text: str


class ProcessResponse(BaseModel):
    """Найденные фрагменты и сведения о модели, выполнившей предсказание"""

    model: ModelInfo
    spans: list[TextSpan]


class TaskAccepted(BaseModel):
    """Задача принята в очередь, адрес результата приходит в заголовке Location"""

    task_id: UUID
    status: TaskStatus


class TaskResponse(BaseModel):
    """Состояние задачи, найденные фрагменты и модель, которая их нашла"""

    task_id: UUID
    status: TaskStatus
    attempts: int
    spans: list[TextSpan] | None
    model_name: str | None
    model_version: str | None
    error: str | None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ErrorResponse(BaseModel):
    """Причина отказа"""

    detail: str
