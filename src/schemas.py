"""Контракты ответов API на Pydantic"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


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


class ProcessRequest(BaseModel):
    """Текст для поиска токсичных фрагментов"""

    text: str = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Отклонить пустое содержимое, сохранив исходные пробелы и позиции"""
        if not value.strip():
            raise ValueError("Текст не должен состоять только из пробелов")
        return value


class ModelInfo(BaseModel):
    """Версия загруженной модели и alias на момент старта приложения"""

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
