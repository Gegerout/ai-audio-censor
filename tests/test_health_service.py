"""Тесты проверки зависимостей без HTTP"""

import asyncpg
import pytest

from src.config import Settings, get_settings
from src.schemas import HealthStatus
from src.services.health import build_report, check_postgres, check_rabbitmq
from tests.fakes import FakeBroker, FakePool


@pytest.fixture
def settings() -> Settings:
    """Настройки из окружения, заданного в conftest"""
    return get_settings()


async def test_check_postgres_returns_version_and_latency(settings) -> None:
    """Если БД отвечает, проверка отдаёт ok, версию и время ответа"""
    component = await check_postgres(FakePool(version="17.0"), settings)

    assert component.status is HealthStatus.OK
    assert component.version == "17.0"
    assert component.latency_ms >= 0
    assert component.error is None


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError(),
        asyncpg.InvalidPasswordError("bad password"),
        asyncpg.InterfaceError("pool is closed"),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_check_postgres_reports_failure(settings, error) -> None:
    """Сетевая ошибка, отказ сервера или закрытый пул дают fail с именем исключения"""
    component = await check_postgres(FakePool(error=error), settings)

    assert component.status is HealthStatus.FAIL
    assert component.version is None
    assert component.error == type(error).__name__


async def test_check_postgres_times_out_when_postgres_hangs(settings) -> None:
    """Зависшая БД не держит проверку дольше таймаута"""
    fast_timeout = settings.model_copy(update={"health_timeout": 0.05})

    component = await check_postgres(FakePool(delay=1), fast_timeout)

    assert component.status is HealthStatus.FAIL
    assert component.error == "TimeoutError"
    assert component.latency_ms < 500


async def test_build_report_ok_when_all_components_ok(settings) -> None:
    """Общий статус ok, если здоровы все зависимости"""
    report = await build_report(FakePool(), FakeBroker(), settings)

    assert report.status is HealthStatus.OK


async def test_build_report_fail_when_any_component_fails(settings) -> None:
    """Общий статус fail, если упала хотя бы одна зависимость"""
    report = await build_report(FakePool(error=OSError()), FakeBroker(), settings)

    assert report.status is HealthStatus.FAIL
    assert report.components["postgres"].status is HealthStatus.FAIL
    assert report.components["rabbitmq"].status is HealthStatus.OK


async def test_check_rabbitmq_reports_connection(settings) -> None:
    """Брокер на связи даёт ok, без связи fail, общий статус тоже fail"""
    alive = await check_rabbitmq(FakeBroker(alive=True), settings)
    report = await build_report(FakePool(), FakeBroker(alive=False), settings)

    assert alive.status is HealthStatus.OK
    assert alive.latency_ms >= 0
    assert report.status is HealthStatus.FAIL
    assert report.components["rabbitmq"].status is HealthStatus.FAIL
    assert report.components["rabbitmq"].error == "ConnectionError"
