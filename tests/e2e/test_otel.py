import os

import pytest
from test_otel_delivery import (
    _attributes,
    _flush,
    _messages_body,
    _metric_points,
    _parse_sse,
    _spans,
    collector as collector,
    post,
    tracing as tracing,
)

from padwan_proxy.proxy import _make_client, build_router

pytestmark = pytest.mark.e2e

_CONFIGURED_ENV = (
    "PADWAN_PROXY_E2E_BASE_URL",
    "PADWAN_PROXY_E2E_MODEL",
)
_API_KEY_ENV = (
    "PADWAN_PROXY_E2E_API_KEY"
    if os.environ.get("PADWAN_PROXY_E2E_API_KEY")
    else "PADWAN_API_KEY"
)


@pytest.mark.skipif(
    any(not os.environ.get(name) for name in (*_CONFIGURED_ENV, _API_KEY_ENV)),
    reason="PADWAN_PROXY_E2E_BASE_URL, MODEL, and API_KEY are required",
)
@pytest.mark.parametrize(
    "tracing", [pytest.param(True, id="content_enabled")], indirect=True
)
async def test_configured_reasoning_stream_exports_otlp(collector, tracing):
    base_url = os.environ["PADWAN_PROXY_E2E_BASE_URL"]
    model = os.environ["PADWAN_PROXY_E2E_MODEL"]
    client = _make_client(base_url, model, _API_KEY_ENV, timeout=300)
    async with client:
        router = build_router(client=client, model=model)
        response = await post(
            router,
            "/v1/messages",
            _messages_body(
                model=model,
                max_tokens=2048,
                stream=True,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Calculate 17 * 23 step by step, then give the result."
                        ),
                    }
                ],
            ),
        )

    assert response.status == 200
    events = _parse_sse(response)
    assert all(name != "error" for name, _ in events)
    assert events[0][0] == "message_start"
    assert events[-1][0] == "message_stop"
    thoughts = "".join(
        event["delta"]["thinking"]
        for _, event in events
        if event.get("delta", {}).get("type") == "thinking_delta"
    )
    answers = "".join(
        event["delta"]["text"]
        for _, event in events
        if event.get("delta", {}).get("type") == "text_delta"
    )
    assert answers.strip(), "reasoning stream returned no final answer deltas"
    assert "391" in answers

    await _flush(tracing)
    [(_, span)] = list(_spans(collector))
    attributes = _attributes(span.attributes)
    assert attributes["gen_ai.request.model"] == model
    assert attributes["gen_ai.usage.input_tokens"] > 0
    assert attributes["gen_ai.usage.output_tokens"] > 0
    assert "gen_ai.input.messages" in attributes
    assert "gen_ai.output.messages" in attributes
    assert "391" in attributes["gen_ai.output.messages"]

    token_points = list(_metric_points(collector, "gen_ai.client.token.usage"))
    token_usage: dict[str, float] = {}
    for point in token_points:
        point_attributes = _attributes(point.attributes)
        assert point_attributes["gen_ai.request.model"] == model
        token_usage[point_attributes["gen_ai.token.type"]] = point.sum
    assert token_usage["input"] > 0
    assert token_usage["output"] > 0
    assert thoughts.strip(), "reasoning stream returned no thinking deltas"
