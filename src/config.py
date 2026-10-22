"""Настройки приложения из переменных окружения и версия из pyproject.toml"""

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


class Settings(BaseSettings):
    """Настройки из переменных окружения и файла .env, имя переменной совпадает с именем поля"""

    # Пустая переменная считается незаданной, так compose передаёт необязательные настройки воркера
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    log_level: str = "INFO"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    # Сколько секунд ждать ответа каждой зависимости в /api/v1/health
    # Только больше нуля, иначе проверка упадёт по таймауту даже при живой БД
    health_timeout: float = Field(default=2.0, gt=0)
    mlflow_tracking_uri: str = Field(default="http://localhost:5001", min_length=1)
    mlflow_model_name: str = Field(default="toxic-spans-detector", min_length=1)
    mlflow_model_alias: str = Field(default="champion", min_length=1)
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_management_port: int = 15672
    rabbitmq_user: str
    rabbitmq_password: SecretStr
    # Сколько задач может ждать в очереди, при заполнении брокер отказывает в публикации
    tasks_queue_limit: int = Field(default=100, gt=0)
    # Размер пула потоков воркера, столько задач он выполняет одновременно
    worker_concurrency: int = Field(default=4, gt=0)
    # Сколько неподтверждённых сообщений брокер отдаёт воркеру, по умолчанию равно размеру пула
    worker_prefetch: int | None = Field(default=None, gt=0)
    # Длительность inference одной задачи в секундах, делает условия замеров воспроизводимыми
    inference_seconds: float = Field(default=0.2, ge=0)
    # Сколько раз задача доставляется воркеру, после этого она уходит в DLQ
    task_max_deliveries: int = Field(default=3, gt=0)
    # Пауза перед возвратом задачи в очередь после ошибки, чтобы не крутить повторы вхолостую
    task_retry_delay_seconds: float = Field(default=1.0, ge=0)
    # Сколько воркер ждёт завершения уже принятых задач при остановке
    worker_graceful_timeout: float = Field(default=30.0, gt=0)
    # Первая пауза источника после отказа брокера, за это время воркер освобождает одно место
    source_initial_delay_seconds: float = Field(default=0.05, gt=0)
    # Предельная пауза источника, за это время воркер разбирает полную очередь
    source_max_delay_seconds: float = Field(default=5.0, gt=0)
    # Как часто источник пишет в лог скорость публикации
    source_report_seconds: float = Field(default=1.0, gt=0)

    @property
    def worker_prefetch_count(self) -> int:
        """Вернуть prefetch_count воркера, без явного значения он равен размеру пула"""
        return self.worker_prefetch or self.worker_concurrency


@lru_cache
def get_settings() -> Settings:
    """Вернуть настройки"""
    return Settings()


@lru_cache
def get_app_version() -> str:
    """Вернуть версию приложения из pyproject.toml"""
    with PYPROJECT_PATH.open("rb") as file:
        return tomllib.load(file)["project"]["version"]
