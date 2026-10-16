"""Обработка текста моделью из MLflow"""

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from src.schemas import ProcessRequest, ProcessResponse

router = APIRouter()


@router.post("/process")
async def process(body: ProcessRequest, request: Request) -> ProcessResponse:
    """Найти токсичные фрагменты, не блокируя цикл обработки запросов"""
    return await run_in_threadpool(request.app.state.inference.process, body.text)
