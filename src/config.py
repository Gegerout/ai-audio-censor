"""Настройки приложения из переменных окружения и версия из pyproject.toml"""

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


class Settings(BaseSettings):
    """Настройки из переменных окружения и файла .env, имя переменной совпадает с именем поля"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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


@lru_cache
def get_settings() -> Settings:
    """Вернуть настройки"""
    return Settings()


@lru_cache
def get_app_version() -> str:
    """Вернуть версию приложения из pyproject.toml"""
    with PYPROJECT_PATH.open("rb") as file:
        return tomllib.load(file)["project"]["version"]
