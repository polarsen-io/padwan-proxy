import asyncio
import atexit
import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.metrics.v1.metrics_pb2 import HistogramDataPoint
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status
from padwan_ai import otel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from test_proxy import (
    COMPLETION,
    TEXT_CHUNKS,
    _messages_body,
    _parse_sse,
    _serve,
    post,
    proxy as proxy,
)

from padwan_proxy.trace import enable_tracing

USAGE = {
    "prompt_tokens": 13,
    "completion_tokens": 8,
    "total_tokens": 21,
    "prompt_tokens_details": {"cached_tokens": 3},
    "completion_tokens_details": {"reasoning_tokens": 5},
}


@dataclass
class OtlpCollector:
    traces: list[ExportTraceServiceRequest]
    metrics: list[ExportMetricsServiceRequest]

    def app(self) -> Starlette:
        async def traces(request: Request) -> Response:
            payload = ExportTraceServiceRequest()
            payload.ParseFromString(await request.body())
            self.traces.append(payload)
            return Response(content=b"", media_type="application/x-protobuf")

        async def metrics(request: Request) -> Response:
            payload = ExportMetricsServiceRequest()
            payload.ParseFromString(await request.body())
            self.metrics.append(payload)
            return Response(content=b"", media_type="application/x-protobuf")

        return Starlette(
            routes=[
                Route("/v1/traces", traces, methods=["POST"]),
                Route("/v1/metrics", metrics, methods=["POST"]),
            ]
        )


@pytest.fixture
async def collector(monkeypatch) -> AsyncIterator[OtlpCollector]:
    collector = OtlpCollector(traces=[], metrics=[])
    server, task, port = await _serve(collector.app())
    for name in tuple(os.environ):
        if name.startswith("OTEL_EXPORTER_OTLP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{port}")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    yield collector
    server.should_exit = True
    await task


@pytest.fixture
async def tracing(request, monkeypatch, collector) -> AsyncIterator[list[Any]]:
    callbacks: list[Any] = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    otel.uninstrument()
    original_output_message = otel._output_message
    original_raw_choice = otel._RawChoice
    enable_tracing(capture_content=getattr(request, "param", False))
    providers = list(
        dict.fromkeys(
            provider
            for callback in callbacks
            if (provider := getattr(callback, "__self__", None)) is not None
            and hasattr(provider, "force_flush")
        )
    )
    yield providers
    otel.uninstrument()
    assert otel._output_message is original_output_message
    assert otel._RawChoice is original_raw_choice
    for provider in providers:
        await asyncio.to_thread(provider.shutdown)


async def _flush(providers: list[Any]) -> None:
    for provider in providers:
        assert await asyncio.to_thread(provider.force_flush)


def _value(value: AnyValue) -> Any:
    field = value.WhichOneof("value")
    if field == "array_value":
        return [_value(item) for item in value.array_value.values]
    if field == "kvlist_value":
        return _attributes(value.kvlist_value.values)
    return getattr(value, field) if field is not None else None


def _attributes(attributes: Iterable[KeyValue]) -> dict[str, Any]:
    return {item.key: _value(item.value) for item in attributes}


def _spans(collector: OtlpCollector) -> Iterator[tuple[dict[str, Any], Span]]:
    for request in collector.traces:
        for resource_spans in request.resource_spans:
            resource = _attributes(resource_spans.resource.attributes)
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    yield resource, span


def _metric_points(collector: OtlpCollector, name: str) -> Iterator[HistogramDataPoint]:
    for request in collector.metrics:
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == name:
                        yield from metric.histogram.data_points


@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
async def test_proxy_delivers_otlp_spans_and_metrics(proxy, collector, tracing, stream):
    backend, router = proxy
    backend.completion = {**COMPLETION, "usage": USAGE}
    backend.stream_chunks = [*TEXT_CHUNKS[:-1], {"choices": [], "usage": USAGE}]

    response = await post(router, "/v1/messages", _messages_body(stream=stream))
    assert response.status == 200
    await _flush(tracing)

    [(resource, span)] = list(_spans(collector))
    attributes = _attributes(span.attributes)
    assert resource["service.name"] == "padwan-proxy"
    assert span.name == "chat glm-4.6"
    assert attributes["gen_ai.request.model"] == "glm-4.6"
    assert attributes.get("gen_ai.request.stream", False) is stream
    assert attributes["gen_ai.usage.input_tokens"] == 13
    assert attributes["gen_ai.usage.output_tokens"] == 8
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 3
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 5

    token_points = list(_metric_points(collector, "gen_ai.client.token.usage"))
    token_usage = {
        _attributes(point.attributes)["gen_ai.token.type"]: point.sum
        for point in token_points
    }
    assert token_usage == {"input": 13, "output": 8}


@pytest.mark.parametrize(
    ("tracing", "captured"),
    [
        pytest.param(False, False, id="content_disabled"),
        pytest.param(True, True, id="content_enabled"),
    ],
    indirect=["tracing"],
)
async def test_proxy_respects_otlp_content_capture(proxy, collector, tracing, captured):
    _, router = proxy
    response = await post(router, "/v1/messages", _messages_body())
    assert response.status == 200
    await _flush(tracing)

    [(_, span)] = list(_spans(collector))
    attributes = _attributes(span.attributes)
    assert ("gen_ai.input.messages" in attributes) is captured
    assert ("gen_ai.output.messages" in attributes) is captured
    if captured:
        assert json.loads(attributes["gen_ai.input.messages"])[0]["parts"] == [
            {"type": "text", "content": "hello"}
        ]
        assert json.loads(attributes["gen_ai.output.messages"])[0]["parts"] == [
            {"type": "text", "content": "Hello!"}
        ]


async def test_proxy_delivers_otlp_error_span(proxy, collector, tracing):
    backend, router = proxy
    backend.status_code = 400

    response = await post(router, "/v1/messages", _messages_body())
    assert response.status == 502
    await _flush(tracing)

    [(_, span)] = list(_spans(collector))
    attributes = _attributes(span.attributes)
    assert span.status.code == Status.STATUS_CODE_ERROR
    assert attributes["error.type"] == "LLMError"


@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
@pytest.mark.parametrize(
    "tracing", [pytest.param(True, id="content_enabled")], indirect=True
)
async def test_structured_thinking_content_is_delivered_as_plain_output(
    proxy, collector, tracing, stream
):
    backend, router = proxy
    thinking = {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "Seven groups of eight."}],
    }
    answer = {"type": "text", "text": "56"}
    backend.completion = {
        **COMPLETION,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": [thinking, answer]},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }
    backend.stream_chunks = [
        {"choices": [{"index": 0, "delta": {"content": [thinking]}}]},
        {"choices": [{"index": 0, "delta": {"content": [answer]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": USAGE},
    ]

    response = await post(
        router,
        "/v1/messages",
        _messages_body(
            stream=stream,
            thinking={"type": "enabled", "budget_tokens": 1024},
            max_tokens=2048,
        ),
    )
    assert response.status == 200
    if stream:
        assert all(name != "error" for name, _ in _parse_sse(response))
    await _flush(tracing)

    [(_, span)] = list(_spans(collector))
    output = json.loads(_attributes(span.attributes)["gen_ai.output.messages"])
    assert output[0]["parts"] == [{"type": "text", "content": "56"}]
