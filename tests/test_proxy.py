import asyncio
import json
import logging
import re
from typing import Any

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

    def app(self) -> Starlette:
        async def chat_completions(request: Request) -> Response:
            body: dict[str, Any] = await request.json()
            self.last_body = body
            if self.status_code != 200:
                return JSONResponse(
                    {"error": {"message": "bad request"}},
                    status_code=self.status_code,
                )
            if body.get("stream"):
                lines = [
                    f"data: {json.dumps(chunk)}\n\n" for chunk in self.stream_chunks
                ]
                lines.append("data: [DONE]\n\n")

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
    with caplog.at_level(logging.INFO, logger="padwan_proxy.proxy"):
        await post(router, "/v1/messages", _messages_body(**body_extra))
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy.proxy"]
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
    with caplog.at_level(logging.INFO, logger="padwan_proxy.proxy"):
        await post(router, "/v1/messages", _messages_body())
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy.proxy"]
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
