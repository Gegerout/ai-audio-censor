"""Эндпоинты API версии 1"""

from fastapi import APIRouter, Request, Response, status

from src.config import get_app_version
from src.schemas import HealthResponse, HealthStatus, VersionResponse
from src.services.health import build_report

router = APIRouter(prefix="/api/v1")


@router.get("/version")
async def version() -> VersionResponse:
    """Вернуть версию приложения из pyproject.toml"""
    return VersionResponse(version=get_app_version())


@router.get("/health")
async def health(request: Request, response: Response) -> HealthResponse:
    """Вернуть состояние зависимостей, код 200 если всё здорово, иначе 503"""
    state = request.app.state
    report = await build_report(state.db_pool, state.broker, state.settings)
    if report.status is HealthStatus.FAIL:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
