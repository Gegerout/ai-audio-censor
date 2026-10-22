"""Тесты настроек, обязательного пароля и проверки значений"""

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_require_postgres_password(monkeypatch) -> None:
    """Без пароля приложение не стартует, значения по умолчанию у него нет"""
    monkeypatch.delenv("POSTGRES_PASSWORD")

    # Локальный .env отключён, иначе пароль подтянулся бы из него
    with pytest.raises(ValidationError, match="postgres_password"):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_settings_reject_non_positive_health_timeout(monkeypatch, value) -> None:
    """Таймаут ноль или меньше отклоняется при старте"""
    monkeypatch.setenv("HEALTH_TIMEOUT", value)

    with pytest.raises(ValidationError, match="health_timeout"):
        Settings(_env_file=None)


def test_settings_hide_password(monkeypatch) -> None:
    """Пароль не виден при выводе настроек, но доступен явным вызовом"""
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")

    settings = Settings(_env_file=None)

    assert "s3cret" not in repr(settings)
    assert settings.postgres_password.get_secret_value() == "s3cret"


@pytest.mark.parametrize(("prefetch", "expected"), [(None, 4), ("", 4), ("10", 10)])
def test_prefetch_defaults_to_pool_size(monkeypatch, prefetch, expected) -> None:
    """Без явного prefetch он равен размеру пула, пустое значение считается незаданным"""
    monkeypatch.setenv("WORKER_CONCURRENCY", "4")
    if prefetch is not None:
        monkeypatch.setenv("WORKER_PREFETCH", prefetch)

    assert Settings(_env_file=None).worker_prefetch_count == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WORKER_CONCURRENCY", "0"),
        ("WORKER_PREFETCH", "0"),
        ("INFERENCE_SECONDS", "-1"),
        ("TASK_MAX_DELIVERIES", "0"),
        ("TASK_RETRY_DELAY_SECONDS", "-1"),
        ("WORKER_GRACEFUL_TIMEOUT", "0"),
    ],
)
def test_settings_reject_invalid_worker_values(monkeypatch, name, value) -> None:
    """Недопустимые настройки воркера отклоняются при старте, а не ломают обработку"""
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.lower()):
        Settings(_env_file=None)
