"""Метрики конвейера и периодическая отправка измерений по OTLP/HTTP"""

import os
from functools import lru_cache

import psutil
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

from src.config import get_telemetry_resource

SCOPE = "ai-audio-censor"
meter = metrics.get_meter(SCOPE)

requests = meter.create_counter("pipeline_api_requests")
api_duration = meter.create_histogram(
    "pipeline_api_duration",
    unit="s",
    explicit_bucket_boundaries_advisory=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
published = meter.create_counter("pipeline_source_published")
refused = meter.create_counter("pipeline_source_refused")
wait_seconds = meter.create_counter("pipeline_source_wait", unit="s")
processed = meter.create_counter("pipeline_worker_processed")
redeliveries = meter.create_counter("pipeline_worker_redeliveries")
in_flight = meter.create_up_down_counter("pipeline_worker_inflight")
inference_duration = meter.create_histogram("pipeline_inference_duration", unit="s")
task_duration = meter.create_histogram(
    "pipeline_task_duration",
    unit="s",
    explicit_bucket_boundaries_advisory=(0.1, 0.25, 0.5, 1, 2, 3, 5, 6, 10, 30, 60, 120, 180),
)


@lru_cache
def configure(service: str, inference_seconds: float) -> None:
    """Настроить экспорт, бакеты модели и сбор RSS в потоке чтения метрик"""
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    if not (
        os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return
    seconds = max(inference_seconds, 0.005)
    bounds = [round(seconds * n, 9) for n in (0.25, 0.5, 0.9, 1, 1.05, 1.25, 2, 5, 10)]
    provider = MeterProvider(
        resource=get_telemetry_resource(service),
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
        views=[
            View(
                instrument_name="pipeline_inference_duration",
                aggregation=ExplicitBucketHistogramAggregation(bounds),
            )
        ],
    )
    metrics.set_meter_provider(provider)
    process = psutil.Process()
    provider.get_meter(SCOPE).create_observable_gauge(
        "process.memory.usage",
        unit="By",
        callbacks=[lambda _: [Observation(process.memory_info().rss)]],
    )
