"""Проверки экспорта метрик, бакетов и меток HTTP"""

import socket
from unittest.mock import Mock, call
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from faststream.rabbit import TestRabbitBroker
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricExportResult

from src import metrics
from src.app import app
from src.broker import observe_queue_limit


@pytest.mark.parametrize(("seconds", "expected"), [(0, 0.005), (0.2, 0.2), (0.8, 0.8)])
def test_metric_provider_uses_configured_buckets_and_rss(monkeypatch, seconds, expected) -> None:
    providers = []
    reader = InMemoryMetricReader()
    monkeypatch.setattr(metrics.metrics, "set_meter_provider", providers.append)
    monkeypatch.setattr(metrics, "PeriodicExportingMetricReader", lambda _: reader)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:4318/v1/metrics")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    configure = metrics.configure.__wrapped__
    configure("worker", seconds)
    assert not providers

    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    configure("worker", seconds)
    (provider,) = providers
    try:
        histogram = provider.get_meter(metrics.SCOPE).create_histogram(
            "pipeline_inference_duration"
        )
        histogram.record(expected)
        (resource,) = reader.get_metrics_data().resource_metrics
        (scope,) = resource.scope_metrics
        points = {m.name: m.data.data_points[0] for m in scope.metrics}
        point = points["pipeline_inference_duration"]
        assert point.explicit_bounds[3] == expected
        assert point.bucket_counts[3] == point.count == 1
        assert points["process.memory.usage"].value > 0
    finally:
        provider.shutdown()


def test_compose_reader_exports_accumulated_count_after_profile_starts(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.setenv("COMPOSE_OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_INTERVAL", "60000")
    providers = []
    monkeypatch.setattr(metrics.metrics, "set_meter_provider", providers.append)
    metrics.configure.__wrapped__("source", 0.2)
    (provider,) = providers
    send = Mock(return_value=MetricExportResult.SUCCESS)
    monkeypatch.setattr(metrics.OTLPMetricExporter, "export", send)
    lookup = Mock(side_effect=socket.gaierror())
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    try:
        counter = provider.get_meter(metrics.SCOPE).create_counter("published")
        counter.add(1)
        provider.force_flush()
        send.assert_not_called()
        lookup.side_effect = None
        counter.add(2)
        provider.force_flush()
        (data,) = send.call_args.args
        (resource,) = data.resource_metrics
        (scope,) = resource.scope_metrics
        (published,) = [m for m in scope.metrics if m.name == "published"]
        assert published.data.data_points[0].value == 3
        lookup.assert_called_with("collector", None, type=socket.SOCK_STREAM)
    finally:
        provider.shutdown()


async def test_api_registers_limit_and_initial_request_counts(monkeypatch) -> None:
    meter = Mock()
    requests = Mock()
    monkeypatch.setattr(metrics, "configure", Mock(return_value=meter))
    monkeypatch.setattr(metrics, "requests", requests)
    async with TestRabbitBroker(app.state.broker, connect_only=True), LifespanManager(app):
        (name,), kwargs = meter.create_observable_gauge.call_args
        (callback,) = kwargs["callbacks"]
        assert name == "pipeline_queue_limit"
        assert callback.func is observe_queue_limit
        assert callback.args == (app.state.settings,)
        assert requests.add.call_args_list == [
            call(0, {"route": "/api/v1/tasks", "status": 202}),
            call(0, {"route": "/api/v1/tasks", "status": 503}),
        ]


async def test_http_labels_use_route_template_and_status(client, monkeypatch) -> None:
    requests = Mock()
    duration = Mock()
    monkeypatch.setattr(metrics, "requests", requests)
    monkeypatch.setattr(metrics, "api_duration", duration)
    for _ in range(2):
        response = await client.get(f"/api/v1/tasks/{uuid4()}")
        assert response.status_code == 503
    assert [call.args for call in requests.add.call_args_list] == [
        (1, {"route": "/api/v1/tasks/{task_id}", "status": 503})
    ] * 2
    assert duration.record.call_count == 2
    assert all(
        seconds > 0 and labels == {"route": "/api/v1/tasks/{task_id}"}
        for (seconds, labels), _ in duration.record.call_args_list
    )
    assert (await client.get("/any/unknown/path")).status_code == 404
    requests.add.assert_called_with(1, {"route": "unmatched", "status": 404})
