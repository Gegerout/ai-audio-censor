"""Проверки настройки SDK и границы контекста задачи"""

from unittest.mock import AsyncMock
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from src import tracing
from src.config import get_settings
from src.source import Source, SourceTask


def test_sdk_respects_environment_and_disabled_export(monkeypatch) -> None:
    providers = []
    monkeypatch.setattr(trace, "set_tracer_provider", providers.append)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    configure = tracing.configure.__wrapped__
    configure("api")
    assert not providers
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.version=test,deployment.environment.name=ci,service.instance.id=fixture",
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-worker")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    configure("worker")
    (provider,) = providers
    try:
        values = provider.resource.attributes
        assert (values["service.name"], values["service.version"]) == ("test-worker", "test")
        assert (values["deployment.environment.name"], values["service.instance.id"]) == (
            "ci",
            "fixture",
        )
        assert provider.sampler is ALWAYS_OFF
    finally:
        provider.shutdown()


async def test_batch_rows_start_independent_traces_with_amqp_headers(spans) -> None:
    broker = AsyncMock()
    tasks = [SourceTask(uuid4(), "first", 0), SourceTask(uuid4(), "second", 1)]
    with tracing.span("outer"):
        await Source(broker, get_settings()).run(tasks, batch="demo")
    roots = [s for s in spans.get_finished_spans() if s.name == "source task"]
    assert len(roots) == len({s.context.trace_id for s in roots}) == 2
    for task, root, call in zip(tasks, roots, broker.publish.await_args_list, strict=True):
        assert root.parent is None
        assert (root.attributes["task.id"], root.attributes["batch.name"]) == (
            str(task.task_id),
            "demo",
        )
        headers = call.kwargs["headers"]
        assert headers["batch"] == "demo"
        context = tracing.extract_context({"traceparent": headers["traceparent"].encode()})
        remote = trace.get_current_span(context).get_span_context()
        assert remote.trace_id == root.context.trace_id
        assert any(
            s.name == "publish"
            and s.context.span_id == remote.span_id
            and s.parent.span_id == root.context.span_id
            for s in spans.get_finished_spans()
        )
    assert (
        not trace.get_current_span(tracing.extract_context({"traceparent": 42}))
        .get_span_context()
        .is_valid
    )
