"""Трассировка конвейера и экспорт спанов в OTLP по HTTP"""

import os
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import Attributes

from src.config import get_app_version

tracer = trace.get_tracer("ai-audio-censor")
propagator = TraceContextTextMapPropagator()


def extract_context(headers: Mapping[str, object]) -> Context:
    """Прочитать W3C контекст из HTTP или типизированных заголовков AMQP"""
    carrier = {}
    for key in ("traceparent", "tracestate"):
        value = headers.get(key)
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        if isinstance(value, str):
            carrier[key] = value
    return propagator.extract(carrier)


@lru_cache
def configure(service: str) -> None:
    """Включить фоновый экспорт при заданном адресе Collector"""
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    if not (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return
    defaults = Resource(
        {
            "service.version": get_app_version(),
            "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT") or "local",
        }
    )
    resource = defaults.merge(
        Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME") or service})
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def record_error(exc: Exception) -> None:
    """Записать тип ошибки и стек без входного текста и параметров запроса"""
    current = trace.get_current_span()
    if current.is_recording():
        name = type(exc).__name__
        current.set_status(trace.StatusCode.ERROR, name)
        current.record_exception(
            exc,
            attributes={
                "exception.message": name,
                "exception.stacktrace": "".join(traceback.format_tb(exc.__traceback__)) + name,
            },
        )


@contextmanager
def span(
    name: str,
    *,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    context: Context | None = None,
    attributes: Attributes = None,
) -> Iterator[trace.Span]:
    """Открыть спан и скрыть чувствительные данные при записи исключения"""
    with tracer.start_as_current_span(
        name,
        kind=kind,
        context=context,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            record_error(cause if isinstance(cause, Exception) else exc)
            raise
