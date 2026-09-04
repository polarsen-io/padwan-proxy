import asyncio
import json
import logging
import re
from typing import Any, cast

import pytest
import uvicorn
from gravier.testing import FakeProto, FakeScope
from padwan_llm.openai.client import OpenAIClient
from piou import CommandError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from padwan_proxy.proxy import _make_client, build_router

TEXT_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
    {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    },
]

TOOL_CHUNKS = [
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"city": "Paris"}'}}
                    ]
                },
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "glm-4.6",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


class FakeBackend:
    """OpenAI-compatible /chat/completions stub recording the last request body."""

    def __init__(self) -> None:
        self.last_body: dict[str, Any] | None = None
        self.stream_chunks: list[dict[str, Any]] = TEXT_CHUNKS
        self.status_code = 200
        self.calls = 0
        self.fail_first = 0  # first N calls answer fail_status instead of streaming
        self.fail_status = 500
        self.malformed_after: int | None = None  # break the SSE stream after N chunks
        self.malformed_calls: int | None = None  # ...on the first N calls (default all)

    def app(self) -> Starlette:
        async def chat_completions(request: Request) -> Response:
            body: dict[str, Any] = await request.json()
            self.last_body = body
            self.calls += 1
            if self.status_code != 200:
                return JSONResponse(
                    {"error": {"message": "bad request"}},
                    status_code=self.status_code,
                )
            if self.calls <= self.fail_first:
                return JSONResponse(
                    {"error": {"message": "backend down"}},
                    status_code=self.fail_status,
                )
            if body.get("stream"):
                broken = self.malformed_after is not None and (
                    self.malformed_calls is None or self.calls <= self.malformed_calls
                )
                chunks = self.stream_chunks
                if broken:
                    chunks = chunks[: self.malformed_after]
                lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
                lines.append("data: {not json\n\n" if broken else "data: [DONE]\n\n")

                async def _gen():
                    for line in lines:
                        yield line

                return StreamingResponse(_gen(), media_type="text/event-stream")
            return JSONResponse(COMPLETION)

        return Starlette(
            routes=[Route("/chat/completions", chat_completions, methods=["POST"])]
        )


async def _serve(app: Starlette) -> tuple[uvicorn.Server, asyncio.Task, int]:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    )
    task = asyncio.ensure_future(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


@pytest.fixture
async def proxy():
    """(FakeBackend, proxy Router) with the fake backend running and client open."""
    backend = FakeBackend()
    backend_server, backend_task, backend_port = await _serve(backend.app())
    client = OpenAIClient(
        model=None,
        base_url=f"http://127.0.0.1:{backend_port}/",
        api_key="test-key",
        # the (connect, read) shape _make_client uses
        timeout=cast(float, (10.0, 3600.0)),
    )
    async with client:
        router = build_router(
            client=client,
            model="glm-4.6",
            small_model="glm-small",
            vision_model="pixtral-test",
            timings=True,
        )
        yield backend, router
    backend_server.should_exit = True
    await backend_task


def _messages_body(**overrides) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
        **overrides,
    }


async def post(router, path: str, body: dict[str, Any]) -> FakeProto:
    proto = FakeProto(json.dumps(body).encode())
    scope = FakeScope(method="POST", path=path)
    await router.dispatch(scope, proto)  # type: ignore[arg-type]
    return proto


def _parse_sse(proto: FakeProto) -> list[tuple[str, dict[str, Any]]]:
    if proto.stream is None:
        raise AssertionError(f"no stream started (status={proto.status})")
    text = b"".join(proto.stream.chunks).decode()
    events = []
    for block in text.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((name, data))
    return events


async def test_messages_non_stream(proxy):
    backend, router = proxy
    proto = await post(router, "/v1/messages", _messages_body())
    assert proto.status == 200
    data = json.loads(proto.body)
    assert data["role"] == "assistant"
    assert data["model"] == "claude-sonnet-5"
    assert data["content"] == [{"type": "text", "text": "Hello!"}]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert backend.last_body["model"] == "glm-4.6"
    assert backend.last_body["messages"] == [{"role": "user", "content": "hello"}]


async def test_messages_stream_text(proxy):
    backend, router = proxy
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    assert proto.status == 200
    assert ("content-type", "text/event-stream") in proto.headers
    events = _parse_sse(proto)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    text = "".join(
        e["delta"]["text"] for _, e in events if e["type"] == "content_block_delta"
    )
    assert text == "Hello"
    assert events[-2][1]["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert backend.last_body["stream_options"] == {"include_usage": True}


async def test_messages_stream_tool_call(proxy):
    backend, router = proxy
    backend.stream_chunks = TOOL_CHUNKS
    body = _messages_body(
        stream=True,
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather.",
                "input_schema": {"type": "object"},
            }
        ],
    )
    proto = await post(router, "/v1/messages", body)
    events = _parse_sse(proto)
    starts = [e for _, e in events if e["type"] == "content_block_start"]
    assert starts == [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": {},
            },
        }
    ]
    partial = "".join(
        e["delta"]["partial_json"]
        for _, e in events
        if e["type"] == "content_block_delta"
    )
    assert json.loads(partial) == {"city": "Paris"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert backend.last_body["tools"][0]["function"]["name"] == "get_weather"


# stream retries


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr("padwan_proxy.proxy._RETRY_BACKOFF", 0.0)


async def test_stream_replayed_when_it_fails_before_any_event(
    proxy, no_backoff, caplog
):
    backend, router = proxy
    backend.malformed_after, backend.malformed_calls = 0, 1
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        proto = await post(router, "/v1/messages", _messages_body(stream=True))
    events = _parse_sse(proto)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    text = "".join(
        e["delta"]["text"] for _, e in events if e["type"] == "content_block_delta"
    )
    assert text == "Hello"  # the failed attempt left nothing behind
    assert backend.calls == 2
    assert any("retrying" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "failure, expected_calls",
    [
        pytest.param({"malformed_after": 0}, 2, id="retries_exhausted"),
        pytest.param(
            {"fail_first": 9, "fail_status": 400}, 1, id="client_error_not_retried"
        ),
        pytest.param(
            {"fail_first": 9, "fail_status": 429}, 1, id="rate_limit_not_retried"
        ),
    ],
)
async def test_stream_error_reported_in_band(
    proxy, no_backoff, failure, expected_calls
):
    backend, router = proxy
    for attr, value in failure.items():
        setattr(backend, attr, value)
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    events = _parse_sse(proto)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["type"] == "error"
    assert backend.calls == expected_calls


async def test_stream_not_replayed_once_events_reached_the_client(proxy, no_backoff):
    backend, router = proxy
    backend.malformed_after = 2
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    names = [name for name, _ in _parse_sse(proto)]
    assert "content_block_delta" in names
    assert names[-1] == "error"
    assert backend.calls == 1


IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "aWNv"},
}


@pytest.mark.parametrize(
    "model, messages, expected_model",
    [
        pytest.param(
            "claude-sonnet-5",
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "look"}, IMAGE_BLOCK],
                }
            ],
            "pixtral-test",
            id="user_image",
        ),
        pytest.param(
            "claude-haiku-4-5",
            [{"role": "user", "content": [IMAGE_BLOCK]}],
            "pixtral-test",
            id="haiku_image_beats_small_model",
        ),
        pytest.param(
            "claude-sonnet-5",
            [
                {"role": "user", "content": "take a screenshot"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "done"}, IMAGE_BLOCK],
                        }
                    ],
                },
            ],
            "pixtral-test",
            id="image_in_tool_result",
        ),
        pytest.param(
            "claude-haiku-4-5",
            [{"role": "user", "content": "no image"}],
            "glm-small",
            id="haiku_routed_to_small_model",
        ),
        pytest.param(
            "claude-sonnet-5",
            [{"role": "user", "content": [{"type": "text", "text": "no image"}]}],
            "glm-4.6",
            id="no_image_keeps_main_model",
        ),
    ],
)
async def test_model_routing(proxy, model, messages, expected_model):
    backend, router = proxy
    await post(router, "/v1/messages", _messages_body(model=model, messages=messages))
    assert backend.last_body["model"] == expected_model


async def test_max_tokens_clamped_to_backend_cap(proxy):
    backend, router = proxy
    await post(router, "/v1/messages", _messages_body(max_tokens=32000))
    assert backend.last_body["max_tokens"] == 16384


async def test_backend_error_mapped_to_anthropic_shape(proxy):
    backend, router = proxy
    backend.status_code = 400
    proto = await post(router, "/v1/messages", _messages_body())
    assert proto.status == 502
    data = json.loads(proto.body)
    assert data["type"] == "error"
    assert data["error"]["type"] == "api_error"


async def test_count_tokens(proxy):
    _, router = proxy
    proto = await post(router, "/v1/messages/count_tokens", _messages_body())
    assert proto.status == 200
    assert json.loads(proto.body)["input_tokens"] > 0


async def test_unknown_route_404(proxy):
    _, router = proxy
    proto = await post(router, "/api/hello", {})
    assert proto.status == 404


# request logging


@pytest.mark.parametrize(
    "body_extra, expected_kind",
    [
        pytest.param({}, "complete", id="non_stream"),
        pytest.param({"stream": True}, "stream", id="stream"),
    ],
)
async def test_requests_logged(proxy, caplog, body_extra, expected_kind):
    _, router = proxy
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", _messages_body(**body_extra))
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    message = record.getMessage()
    assert "claude-sonnet-5 → glm-4.6" in message
    assert expected_kind in message
    assert "end_turn" in message
    assert "in=10 out=2" in message
    # timing split from timings=True: total ≈ backend + req-xlate + proxy overhead
    assert re.search(
        r"\(backend \d+\.\d+s, req-xlate \d+\.\dms, proxy \d+\.\dms\)", message
    )


async def test_backend_error_logged_as_warning(proxy, caplog):
    backend, router = proxy
    backend.status_code = 400
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", _messages_body())
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    assert record.levelno == logging.WARNING
    assert "request failed" in record.getMessage()


# _make_client


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("PADWAN_BASE_URL", "PADWAN_API_KEY", "OPENAI_API_KEY", "MY_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "env, backend_url, api_key_env, expected_url, expected_key",
    [
        pytest.param(
            {"MY_KEY": "sk-explicit"},
            "https://api.example.com/v1/",
            "MY_KEY",
            "https://api.example.com/v1/",
            "sk-explicit",
            id="explicit_env_var",
        ),
        pytest.param(
            {"PADWAN_API_KEY": "sk-gw", "OPENAI_API_KEY": "sk-openai"},
            "https://api.example.com/v1/",
            None,
            "https://api.example.com/v1/",
            "sk-gw",
            id="padwan_key_preferred_over_openai",
        ),
        pytest.param(
            {
                "PADWAN_BASE_URL": "https://gw.example.com/v1/",
                "PADWAN_API_KEY": "sk-gw",
            },
            None,
            None,
            "https://gw.example.com/v1/",
            "sk-gw",
            id="gateway_url_fallback",
        ),
    ],
)
def test_make_client_resolution(
    clean_env, env, backend_url, api_key_env, expected_url, expected_key
):
    for k, v in env.items():
        clean_env.setenv(k, v)
    client = _make_client(backend_url, "glm-4.6", api_key_env)
    assert client.base_url == expected_url
    assert client._api_key == expected_key
    # reasoning models go silent for minutes, but a dead host must fail fast
    assert client.timeout == (10.0, 3600)


@pytest.mark.parametrize(
    "env, backend_url, api_key_env, match",
    [
        pytest.param({}, None, None, "No backend URL", id="no_url"),
        pytest.param(
            {},
            "https://api.example.com/v1/",
            "MY_KEY",
            "MY_KEY not set",
            id="env_unset",
        ),
    ],
)
def test_make_client_errors(clean_env, env, backend_url, api_key_env, match):
    for k, v in env.items():
        clean_env.setenv(k, v)
    with pytest.raises(CommandError) as exc:
        _make_client(backend_url, "glm-4.6", api_key_env)
    assert match in exc.value.message
