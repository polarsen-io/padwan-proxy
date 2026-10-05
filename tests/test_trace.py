from types import SimpleNamespace

import pytest

import padwan_proxy.trace as trace


@pytest.mark.parametrize(
    "endpoint_var",
    [
        pytest.param(None, id="langfuse_only"),
        pytest.param("OTEL_EXPORTER_OTLP_ENDPOINT", id="both"),
        pytest.param("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", id="both_traces_endpoint"),
    ],
)
def test_langfuse_dual_export(monkeypatch, endpoint_var):
    import atexit

    import padwan_ai.langfuse
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    if endpoint_var:
        monkeypatch.setenv(endpoint_var, "http://collector:4318/v1/traces")
    provider = TracerProvider()
    langfuse_exporter = InMemorySpanExporter()
    otlp_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(langfuse_exporter))
    monkeypatch.setattr(
        padwan_ai.langfuse,
        "instrument",
        lambda **kwargs: SimpleNamespace(
            tracer_provider=provider, shutdown=provider.shutdown
        ),
    )
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda: otlp_exporter)
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(trace, "_langfuse_active", False)
    try:
        trace._enable_langfuse(capture_content=False)
        with provider.get_tracer("padwan_ai").start_as_current_span("request"):
            pass
        provider.force_flush()
        spans = langfuse_exporter.get_finished_spans()
        assert len(spans) == 1
        exported = otlp_exporter.get_finished_spans()
        assert len(exported) == (1 if endpoint_var else 0)
        if endpoint_var:
            assert exported[0].context == spans[0].context
    finally:
        for callback in reversed(callbacks):
            callback()
        provider.shutdown()


class TestEnableTracing:
    def test_langfuse_picked_by_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        calls: list[str] = []
        monkeypatch.setattr(
            trace,
            "_enable_langfuse",
            lambda *, capture_content: calls.append("langfuse"),
        )
        monkeypatch.setattr(
            trace,
            "_enable_otlp",
            lambda *, capture_content: calls.append("otlp"),
        )

        trace.enable_tracing()

        assert calls == ["langfuse"]
