"""Проверки экспорта метрик, бакетов и меток HTTP"""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from src import metrics


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
