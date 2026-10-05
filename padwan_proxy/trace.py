from __future__ import annotations

import atexit
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from .utils import console

__all__ = ("enable_tracing", "session_context")

_langfuse_active = False


def _plain_openai_content(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text = "".join(
        value
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(value := part.get("text"), str)
    )
    return text or None


class _CapturedText(list[str]):
    def append(self, content: object) -> None:
        if text := _plain_openai_content(content):
            super().append(text)


def _install_openai_content_compat(otel: Any) -> None:
    # padwan-ai 0.11.1 still assumes raw OpenAI content is always a string.
    install = getattr(otel, "_install", None)
    if not callable(install) or not all(
        hasattr(otel, name) for name in ("_output_message", "_RawChoice")
    ):
        raise RuntimeError(
            "padwan-ai OpenTelemetry internals changed; structured content "
            "compatibility is unavailable"
        )

    def wrap_output_message(original: Any) -> Any:
        def wrapped(
            content: object, tool_calls: object, finish_reason: str | None
        ) -> Any:
            return original(_plain_openai_content(content), tool_calls, finish_reason)

        return wrapped

    def wrap_raw_choice(original: Any) -> Any:
        def wrapped(*args: object, **kwargs: object) -> Any:
            choice = original(*args, **kwargs)
            choice.text = _CapturedText(choice.text)
            return choice

        return wrapped

    install(
        (
            (otel, "_output_message", wrap_output_message),
            (otel, "_RawChoice", wrap_raw_choice),
        )
    )


def _enable_langfuse(*, capture_content: bool) -> None:
    global _langfuse_active
    from padwan_ai import otel
    from padwan_ai.langfuse import instrument

    integration = instrument(capture_content=capture_content)
    otlp_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ) or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    try:
        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            integration.tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter())
            )
        if capture_content:
            _install_openai_content_compat(otel)
    except BaseException:
        integration.shutdown()
        integration.tracer_provider.shutdown()
        raise
    atexit.register(integration.tracer_provider.shutdown)
    atexit.register(integration.shutdown)
    _langfuse_active = True
    destinations = f"Langfuse + OTLP → {otlp_endpoint}" if otlp_endpoint else "Langfuse"
    console.print(f"[dim]Tracing enabled ({destinations})[/dim]")


def session_context(session_id: str | None) -> AbstractContextManager[Any]:
    """Group the spans opened inside under one Langfuse session; a no-op otherwise."""
    if session_id is None or not _langfuse_active:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(session_id=session_id)


def _enable_otlp(*, capture_content: bool) -> None:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from padwan_ai import otel

    resource = Resource.create({"service.name": "padwan-proxy"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    atexit.register(meter_provider.shutdown)
    atexit.register(tracer_provider.shutdown)
    try:
        otel.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            capture_content=capture_content,
        )
        if capture_content:
            _install_openai_content_compat(otel)
    except BaseException:
        otel.uninstrument()
        atexit.unregister(tracer_provider.shutdown)
        atexit.unregister(meter_provider.shutdown)
        tracer_provider.shutdown()
        meter_provider.shutdown()
        raise
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    console.print(f"[dim]Tracing enabled (OTLP → {endpoint})[/dim]")


def enable_tracing(*, capture_content: bool = False) -> None:
    """Instrument padwan-ai clients for this process.

    Exports to Langfuse when `LANGFUSE_PUBLIC_KEY` is set, and also to OTLP
    when an OTLP endpoint is explicit; otherwise defaults to OTLP alone.
    Exporters are flushed at exit.
    `capture_content` also records prompts and completions on the spans.
    """
    try:
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            _enable_langfuse(capture_content=capture_content)
        else:
            _enable_otlp(capture_content=capture_content)
    except ImportError:
        console.print("[red]Tracing dependencies not installed.[/red]")
        console.print("[dim]Install the trace extra: `uv sync --extra trace`.[/dim]")
        raise SystemExit(1)
