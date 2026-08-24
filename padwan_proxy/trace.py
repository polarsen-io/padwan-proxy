from __future__ import annotations

import atexit
import os

from .utils import console

__all__ = ("enable_tracing",)


def _enable_langfuse() -> None:
    from padwan_llm.langfuse import instrument

    integration = instrument()
    atexit.register(integration.shutdown)
    console.print("[dim]Tracing enabled (Langfuse)[/dim]")


def _enable_otlp() -> None:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from padwan_llm import otel

    resource = Resource.create({"service.name": "padwan-proxy"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    otel.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
    atexit.register(meter_provider.shutdown)
    atexit.register(tracer_provider.shutdown)
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    console.print(f"[dim]Tracing enabled (OTLP → {endpoint})[/dim]")


def enable_tracing() -> None:
    """Instrument padwan-llm clients for this process.

    Exports to Langfuse when `LANGFUSE_PUBLIC_KEY` is set (the adapter reads
    the standard `LANGFUSE_*` env vars), otherwise over OTLP using the
    standard `OTEL_EXPORTER_OTLP_*` env vars. Exporters are flushed at exit.
    """
    try:
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            _enable_langfuse()
        else:
            _enable_otlp()
    except ImportError:
        console.print("[red]Tracing dependencies not installed.[/red]")
        console.print("[dim]Install the trace extra: `uv sync --extra trace`.[/dim]")
        raise SystemExit(1)
