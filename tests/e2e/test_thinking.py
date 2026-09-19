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


def test_thinking_non_stream(live_proxy: tuple[str, Provider]) -> None:
    base_url, provider = live_proxy
    content_type, raw = _post(base_url, _body(provider.model, stream=False))
    response = json.loads(raw)
    thoughts = [
        block["thinking"]
        for block in response["content"]
        if block["type"] == "thinking"
    ]
    answers = [
        block["text"] for block in response["content"] if block["type"] == "text"
    ]

    assert content_type == "application/json"
    assert "".join(thoughts).strip(), "reasoning model returned no thinking block"
    assert "".join(answers).strip(), "reasoning model returned no final answer"
    assert "391" in "".join(answers)


def test_thinking_stream(live_proxy: tuple[str, Provider]) -> None:
    base_url, provider = live_proxy
    content_type, raw = _post(base_url, _body(provider.model, stream=True))
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append(
            (
                lines[0].removeprefix("event: "),
                json.loads(lines[1].removeprefix("data: ")),
            )
        )

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

    assert content_type == "text/event-stream"
    assert events[0][0] == "message_start"
    assert events[-1][0] == "message_stop"
    assert thoughts.strip(), "reasoning stream returned no thinking deltas"
    assert answers.strip(), "reasoning stream returned no final answer deltas"
    assert "391" in answers
