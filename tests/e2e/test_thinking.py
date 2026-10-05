import json
from typing import Any
from urllib.request import Request, urlopen

import pytest

from .conftest import Provider

pytestmark = pytest.mark.e2e


def _post(base_url: str, body: dict[str, Any]) -> tuple[str, str]:
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return response.headers.get_content_type(), response.read().decode()


def _body(model: str, *, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            {
                "role": "user",
                "content": "Calculate 17 * 23 step by step, then give the result.",
            }
        ],
        "stream": stream,
    }


def _sse_events(raw: str) -> list[dict[str, Any]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append(json.loads(lines[1].removeprefix("data: ")))
    return events


@pytest.mark.parametrize(
    "stream",
    [pytest.param(False, id="non_stream"), pytest.param(True, id="stream")],
)
def test_thinking(live_proxy: tuple[str, Provider], stream: bool) -> None:
    base_url, provider = live_proxy
    content_type, raw = _post(base_url, _body(provider.model, stream=stream))
    if stream:
        events = _sse_events(raw)
        thoughts = "".join(
            e["delta"]["thinking"]
            for e in events
            if e.get("delta", {}).get("type") == "thinking_delta"
        )
        answers = "".join(
            e["delta"]["text"]
            for e in events
            if e.get("delta", {}).get("type") == "text_delta"
        )
        assert content_type == "text/event-stream"
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
    else:
        content = json.loads(raw)["content"]
        thoughts = "".join(b["thinking"] for b in content if b["type"] == "thinking")
        answers = "".join(b["text"] for b in content if b["type"] == "text")
        assert content_type == "application/json"

    assert thoughts.strip(), "reasoning model returned no thinking"
    assert answers.strip(), "reasoning model returned no final answer"
    assert "391" in answers
