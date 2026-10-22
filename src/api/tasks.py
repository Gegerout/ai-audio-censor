"""Приём задач на обработку и выдача их результата"""

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status

from src.schemas import ErrorResponse, TaskAccepted, TaskRequest, TaskResponse, TaskStatus
from src.services.tasks import STORAGE_ERRORS, TaskRejectedError, submit_task
from src.storage import get_task

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

UNAVAILABLE = {
    "model": ErrorResponse,
    "description": "Задачу сейчас принять нельзя, заголовок Retry-After говорит, когда повторить",
}


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: UNAVAILABLE},
)
async def create_task(body: TaskRequest, request: Request, response: Response) -> TaskAccepted:
    """Поставить текст в очередь, при заполненной очереди сразу ответить 503"""
    state = request.app.state
    try:
        task_id = await submit_task(state.db_pool, state.broker, state.settings, body.text)
    except TaskRejectedError as exc:
        log.warning("task rejected", extra={"reason": exc.reason, "retry_after": exc.retry_after})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    log.info("task accepted", extra={"task_id": str(task_id)})
    response.headers["Location"] = f"{router.prefix}/{task_id}"
    return TaskAccepted(task_id=task_id, status=TaskStatus.PENDING)


@router.get(
    "/{task_id}",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def read_task(task_id: UUID, request: Request) -> TaskResponse:
    """Вернуть состояние задачи и результат, когда воркер его записал"""
    settings = request.app.state.settings
    try:
        async with asyncio.timeout(settings.tasks_timeout_seconds):
            record = await get_task(request.app.state.db_pool, task_id)
    except STORAGE_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="База недоступна"
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")
    return TaskResponse(
        task_id=record.id,
        status=record.status,
        attempts=record.attempts,
        spans=record.result,
        model_name=record.model_name,
        model_version=record.model_version,
        error=record.error,
        enqueued_at=record.enqueued_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )
