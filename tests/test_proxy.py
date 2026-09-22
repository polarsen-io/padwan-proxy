import asyncio
import json
import logging
import re
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import Mock

import pytest
import uvicorn
from granian._granian import RSGIProtocolClosed
from gravier.testing import FakeProto, FakeScope
from padwan_ai.gemini.client import GeminiClient
from padwan_ai.openai.client import OpenAIClient
from piou import CommandError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from padwan_proxy.proxy import (
    _client_session,
    _error_detail,
    _make_client,
    _translate_body,
    _validate_body,
    build_router,
)

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
        self.completion: dict[str, Any] = COMPLETION
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
            return JSONResponse(self.completion)

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
async def proxy(request):
    """(FakeBackend, proxy Router) with the fake backend running and client open.

    An indirect param overrides `build_router` kwargs.
    """
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
            **getattr(request, "param", {}),
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


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            {
                "user_id": json.dumps(
                    {
                        "device_id": "d",
                        "account_uuid": "",
                        "session_id": "a7cade63-3dc7-4cf6-b926-981d7b3df7c7",
                    }
                )
            },
            "a7cade63-3dc7-4cf6-b926-981d7b3df7c7",
            id="json",
        ),
        pytest.param(
            {
                "user_id": "user_abc_account_123"
                "_session_a7cade63-3dc7-4cf6-b926-981d7b3df7c7"
            },
            "a7cade63-3dc7-4cf6-b926-981d7b3df7c7",
            id="legacy",
        ),
        pytest.param({"user_id": json.dumps({"session_id": ""})}, None, id="empty"),
        pytest.param({"user_id": "{not json"}, None, id="malformed"),
        pytest.param({"user_id": "opaque-user"}, None, id="opaque"),
        pytest.param(None, None, id="absent"),
    ],
)
def test_client_session(metadata, expected):
    body = _messages_body()
    if metadata is not None:
        body["metadata"] = metadata
    assert _client_session(body) == expected


@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
async def test_session_propagated_to_tracing(proxy, monkeypatch, stream):
    backend, router = proxy
    seen: list[str | None] = []

    @contextmanager
    def fake_session_context(session_id):
        seen.append(session_id)
        yield

    monkeypatch.setattr("padwan_proxy.proxy.session_context", fake_session_context)
    body = _messages_body(
        stream=stream, metadata={"user_id": json.dumps({"session_id": "sess-1"})}
    )

    proto = await post(router, "/v1/messages", body)

    assert proto.status == 200
    assert seen == ["sess-1"]


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


@pytest.mark.parametrize(
    "reasoning, answer",
    [
        pytest.param(
            {"reasoning_content": "Seven groups of eight."},
            {"content": "56"},
            id="reasoning_content",
        ),
        pytest.param(
            {"reasoning": "Seven groups of eight."},
            {"content": "56"},
            id="scaleway_reasoning",
        ),
        pytest.param(
            {
                "reasoning": "Ignored alias",
                "reasoning_content": "Seven groups of eight.",
            },
            {"content": "56"},
            id="native_reasoning_takes_precedence",
        ),
        pytest.param(
            {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [
                            {"type": "text", "text": "Seven groups of eight."}
                        ],
                    }
                ]
            },
            {"content": [{"type": "text", "text": "56"}]},
            id="structured_thinking",
        ),
    ],
)
@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
async def test_thinking_models_separate_thoughts_and_answer(
    proxy, reasoning, answer, stream
):
    backend, router = proxy
    message = {**reasoning, **answer}
    if isinstance(reasoning.get("content"), list):
        message["content"] = reasoning["content"] + answer["content"]
    backend.completion = {
        **COMPLETION,
        "choices": [{"message": message, "finish_reason": "stop"}],
    }
    backend.stream_chunks = [
        {"choices": [{"index": 0, "delta": delta}]} for delta in (reasoning, answer)
    ] + TEXT_CHUNKS[-2:]
    proto = await post(
        router,
        "/v1/messages",
        _messages_body(
            stream=stream,
            thinking={"type": "enabled", "budget_tokens": 1024},
            max_tokens=2048,
        ),
    )
    assert proto.status == 200
    if stream:
        events = _parse_sse(proto)
        starts = [
            event["content_block"]["type"]
            for name, event in events
            if name == "content_block_start"
        ]
        assert starts == ["thinking", "text"]
        deltas = [
            event["delta"] for name, event in events if name == "content_block_delta"
        ]
        assert deltas == [
            {"type": "thinking_delta", "thinking": "Seven groups of eight."},
            {"type": "text_delta", "text": "56"},
        ]
        assert [
            event["index"] for name, event in events if name == "content_block_stop"
        ] == [0, 1]
        assert events[-2][1]["delta"]["stop_reason"] == "end_turn"
        assert events[-1][0] == "message_stop"
    else:
        response = json.loads(proto.body)
        assert response["content"] == [
            {"type": "thinking", "thinking": "Seven groups of eight."},
            {"type": "text", "text": "56"},
        ]
        assert response["stop_reason"] == "end_turn"


async def test_thinking_tool_turn_can_be_replayed(proxy):
    backend, router = proxy
    backend.stream_chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "Check the weather."}}
            ]
        }
    ] + TOOL_CHUNKS
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    events = _parse_sse(proto)
    assert [
        event["content_block"]["type"]
        for name, event in events
        if name == "content_block_start"
    ] == ["thinking", "tool_use"]
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    followup = _messages_body(
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Check the weather."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}
                ],
            },
        ]
    )
    response = await post(router, "/v1/messages", followup)
    assert response.status == 200
    assert backend.last_body["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sunny",
    }
    assert backend.last_body["messages"][1]["tool_calls"][0]["id"] == "call_1"


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


class _DisconnectTransport:
    """RSGI transport that closes after N sends — simulates a client drop."""

    def __init__(self, after: int) -> None:
        self._after = after
        self._sent = 0

    async def send_str(self, data: str) -> None:
        self._sent += 1
        if self._sent > self._after:
            raise RSGIProtocolClosed("RSGI transport is closed")


async def test_disconnect_before_first_chunk_does_not_replay_backend(proxy, caplog):
    backend, router = proxy
    # zero chunks: first_chunk never flips, so a send drops the held frames with
    # sent=False — the path that, without the inner re-raise, would replay the backend.
    backend.stream_chunks = []
    body = _messages_body(stream=True)

    class _DroppingProto(FakeProto):
        def response_stream(self, status, headers):  # type: ignore[override]
            self.status = status
            self.headers = headers
            self.stream = _DisconnectTransport(after=0)  # type: ignore[assignment]
            return self.stream

    proto = _DroppingProto(json.dumps(body).encode())
    scope = FakeScope(method="POST", path="/v1/messages")
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await router.dispatch(scope, proto)  # type: ignore[arg-type]
    # client already gone: no retry, no in-band error, single backend call
    assert backend.calls == 1
    assert not any("retrying" in r.message for r in caplog.records)
    assert any("client disconnected" in r.message for r in caplog.records)


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
        pytest.param(
            "glm-small",
            [{"role": "user", "content": "no image"}],
            "glm-small",
            id="backend_name_passes_through",
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


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"{", id="malformed_json"),
        pytest.param(b"[]", id="not_an_object"),
        pytest.param(b"{}", id="missing_fields"),
        *[
            pytest.param(json.dumps(_messages_body(**override)).encode(), id=name)
            for name, override in (
                ("tokens_string", {"max_tokens": "100"}),
                ("tokens_bool", {"max_tokens": True}),
                ("tokens_zero", {"max_tokens": 0}),
                ("tokens_negative", {"max_tokens": -1}),
                ("model_empty", {"model": ""}),
                ("messages_empty", {"messages": []}),
                ("message_not_object", {"messages": ["hello"]}),
                ("invalid_role", {"messages": [{"role": "unknown", "content": "hi"}]}),
                (
                    "system_nontext",
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "x",
                                        "content": "hi",
                                    }
                                ],
                            }
                        ]
                    },
                ),
                ("metadata_not_object", {"metadata": "bad"}),
                ("stream_not_bool", {"stream": "true"}),
                ("tool_not_object", {"tools": ["bad"]}),
            )
        ],
    ],
)
async def test_invalid_messages_rejected_before_backend(proxy, raw):
    backend, router = proxy
    proto = FakeProto(raw)
    await router.dispatch(FakeScope(method="POST", path="/v1/messages"), proto)
    assert proto.status == 400
    error = json.loads(proto.body)
    assert error["type"] == "error"
    assert error["error"]["type"] == "invalid_request_error"
    assert backend.calls == 0


async def test_request_extensions_reach_translator(proxy, monkeypatch):
    from padwan_proxy import proxy as proxy_module

    _, router = proxy
    translate = Mock(wraps=proxy_module.messages_to_openai)
    monkeypatch.setattr(proxy_module, "messages_to_openai", translate)
    tool = {
        "name": "custom_tool",
        "input_schema": {"type": "object"},
        "defer_loading": True,
    }
    proto = await post(router, "/v1/messages", _messages_body(tools=[tool]))
    assert proto.status == 200
    assert translate.call_args.args[0]["tools"] == [tool]


@pytest.mark.parametrize(
    "inline_system",
    [
        pytest.param("inline instructions", id="string"),
        pytest.param(
            [{"type": "text", "text": "inline instructions"}], id="text_blocks"
        ),
    ],
)
def test_inline_system_translation_preserves_order_and_tool_result(inline_system):
    body = _messages_body(
        system="top-level instructions",
        messages=[
            {"role": "user", "content": "Read the file."},
            {"role": "system", "content": inline_system},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "fact.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "content": [{"type": "text", "text": "probe-token"}],
                    }
                ],
            },
        ],
    )

    _validate_body(body)
    translated = _translate_body(body, model="glm-5.2")

    assert translated["messages"] == [
        {"role": "system", "content": "top-level instructions"},
        {"role": "user", "content": "Read the file."},
        {"role": "system", "content": "inline instructions"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "read-1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": '{"file_path":"fact.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read-1", "content": "probe-token"},
    ]


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
    "body_extra, expected_kind, expected_route",
    [
        pytest.param({}, "complete", "claude-sonnet-5 → glm-4.6", id="non_stream"),
        pytest.param(
            {"stream": True}, "stream", "claude-sonnet-5 → glm-4.6", id="stream"
        ),
        pytest.param(
            {"model": "glm-small"}, "complete", "glm-small", id="backend_small"
        ),
        pytest.param(
            {"model": "glm-4.6", "stream": True}, "stream", "glm-4.6", id="backend_main"
        ),
    ],
)
async def test_requests_logged(
    proxy, caplog, body_extra, expected_kind, expected_route
):
    _, router = proxy
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", _messages_body(**body_extra))
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    message = record.getMessage()
    assert message.split(" | ", 1)[0] == expected_route
    assert expected_kind in message
    assert "end_turn" in message
    assert "in=10 out=2" in message
    # timing split from timings=True: total ≈ backend + req-xlate + proxy overhead
    assert re.search(
        r"\(backend \d+\.\d+s, req-xlate \d+\.\dms, proxy \d+\.\dms\)", message
    )


@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
async def test_session_logged_in_request_head(proxy, caplog, stream):
    _, router = proxy
    body = _messages_body(
        stream=stream,
        metadata={"user_id": json.dumps({"session_id": "a7cade63-session"})},
    )
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", body)
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    assert record.getMessage().startswith("a7cade63 ")


@pytest.mark.parametrize("proxy", [{"breakdown": True}], indirect=True)
async def test_breakdown_logged(proxy, caplog):
    _, router = proxy
    body = _messages_body(
        system="you are a helpful assistant",
        tools=[
            {"name": "mcp__argent__describe", "input_schema": {"type": "object"}},
            {"name": "Read", "input_schema": {"type": "object"}},
        ],
    )
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", body)
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    first, second = record.getMessage().splitlines()
    assert "in=10 out=2" in first
    assert re.match(r" +sys=\d+ tools=\d+\(2\) msgs=\d+ \| top: ", second)
    assert "argent" in second and "builtin" in second


@pytest.mark.parametrize(
    "stream",
    [pytest.param(True, id="stream"), pytest.param(False, id="complete")],
)
async def test_tool_use_log_names_the_tools(proxy, caplog, stream):
    backend, router = proxy
    backend.stream_chunks = TOOL_CHUNKS
    backend.completion = {
        **COMPLETION,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", _messages_body(stream=stream))
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    assert "tool_use(get_weather)" in record.getMessage()


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(400, id="bad_request"),
        pytest.param(402, id="quota"),
        pytest.param(429, id="rate_limit"),
    ],
)
@pytest.mark.parametrize(
    "stream", [pytest.param(False, id="complete"), pytest.param(True, id="stream")]
)
async def test_backend_error_logged_as_warning(proxy, caplog, status, stream):
    backend, router = proxy
    backend.status_code = status
    with caplog.at_level(logging.INFO, logger="padwan_proxy"):
        await post(router, "/v1/messages", _messages_body(stream=stream))
    (record,) = [r for r in caplog.records if r.name == "padwan_proxy"]
    assert record.levelno == logging.WARNING
    assert ("stream failed" if stream else "request failed") in record.getMessage()
    assert f"HTTP {status}" in record.getMessage()


def test_empty_backend_error_includes_exception_type():
    assert _error_detail(TimeoutError()) == "TimeoutError()"


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
            {
                "MY_KEY": "sk-explicit",
                "PADWAN_API_KEY": "sk-gw",
                "OPENAI_API_KEY": "sk-openai",
            },
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
        pytest.param(
            {"OPENAI_API_KEY": "sk-openai"},
            "https://api.example.com/v1/",
            None,
            "https://api.example.com/v1/",
            "sk-openai",
            id="openai_key_fallback",
        ),
        pytest.param(
            {
                "PADWAN_BASE_URL": "https://gw.example.com/v1/",
                "OPENAI_API_KEY": "sk-openai",
            },
            None,
            None,
            "https://gw.example.com/v1/",
            "sk-openai",
            id="gateway_openai_key_fallback",
        ),
        pytest.param(
            {},
            "https://api.example.com/v1/",
            None,
            "https://api.example.com/v1/",
            "no-key-required",
            id="unauthenticated_backend",
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
    "model, provider",
    [
        pytest.param("gemini-2.5-pro", "gemini", id="gemini_name"),
        pytest.param("claude-sonnet-4", "openai", id="anthropic_name"),
        pytest.param("glm-4.6", "openai", id="openai_compatible_name"),
    ],
)
def test_custom_endpoint_transport_by_model(clean_env, model, provider):
    client = _make_client("https://backend.example/v1/", model, None)
    assert client.base_url == "https://backend.example/v1/"
    assert client.provider == provider
    assert client._api_key == "no-key-required"


@pytest.mark.parametrize(
    "env, backend_url, api_key_env, match",
    [
        pytest.param({}, None, None, "No backend URL", id="no_url"),
        pytest.param(
            {"PADWAN_API_KEY": "sk-gw", "OPENAI_API_KEY": "sk-openai"},
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


# --- Native Gemini backend -------------------------------------------------


GEMINI_USAGE = {
    "promptTokenCount": 10,
    "candidatesTokenCount": 2,
    "totalTokenCount": 12,
}

GEMINI_TEXT_CHUNKS = [
    {
        "candidates": [
            {"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}
        ]
    },
    {"usageMetadata": GEMINI_USAGE},
]

GEMINI_TOOL_CHUNKS = [
    {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Paris"},
                                "id": "call_1",
                            }
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    },
    {"usageMetadata": GEMINI_USAGE},
]

GEMINI_COMPLETION = {
    "candidates": [
        {"content": {"parts": [{"text": "Hello!"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": GEMINI_USAGE,
}


class FakeGeminiBackend:
    """Native Gemini :generateContent / :streamGenerateContent stub."""

    def __init__(self) -> None:
        self.last_body: dict[str, Any] | None = None
        self.stream_chunks: list[dict[str, Any]] = GEMINI_TEXT_CHUNKS
        self.completion: dict[str, Any] = GEMINI_COMPLETION

    def app(self) -> Starlette:
        async def generate(request: Request) -> Response:
            self.last_body = cast("dict[str, Any]", await request.json())
            return JSONResponse(self.completion)

        async def stream_generate(request: Request) -> Response:
            self.last_body = cast("dict[str, Any]", await request.json())
            lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in self.stream_chunks]

            async def _gen():
                for line in lines:
                    yield line

            return StreamingResponse(_gen(), media_type="text/event-stream")

        return Starlette(
            routes=[
                Route(
                    "/models/gemini-2.5-flash:generateContent",
                    generate,
                    methods=["POST"],
                ),
                Route(
                    "/models/gemini-2.5-flash:streamGenerateContent",
                    stream_generate,
                    methods=["POST"],
                ),
            ]
        )


@pytest.fixture
async def gemini_proxy(request):
    """A proxy backed by a native GeminiClient pointed at FakeGeminiBackend."""
    backend = FakeGeminiBackend()
    backend_server, backend_task, backend_port = await _serve(backend.app())
    client = GeminiClient(
        model="gemini-2.5-flash",
        base_url=f"http://127.0.0.1:{backend_port}/",
        api_key="test-key",
        timeout=cast(float, (10.0, 3600.0)),
    )
    async with client:
        router = build_router(
            client=client,
            model="gemini-2.5-flash",
            small_model="gemini-2.5-flash-lite",
            **getattr(request, "param", {}),
        )
        yield backend, router
    backend_server.should_exit = True
    await backend_task


async def test_gemini_non_stream(gemini_proxy):
    backend, router = gemini_proxy
    proto = await post(router, "/v1/messages", _messages_body())
    assert proto.status == 200
    data = json.loads(proto.body)
    assert data["role"] == "assistant"
    assert data["model"] == "claude-sonnet-5"
    assert data["content"] == [{"type": "text", "text": "Hello!"}]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert backend.last_body["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]}
    ]
    assert backend.last_body["generationConfig"]["maxOutputTokens"] == 100
    # Gemini path must not carry OpenAI-only shims.
    assert "stream_options" not in backend.last_body


async def test_gemini_stream_text(gemini_proxy):
    backend, router = gemini_proxy
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    assert proto.status == 200
    events = _parse_sse(proto)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
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


async def test_gemini_stream_tool_call(gemini_proxy):
    backend, router = gemini_proxy
    backend.stream_chunks = GEMINI_TOOL_CHUNKS
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    assert proto.status == 200
    events = _parse_sse(proto)
    start = [e for _, e in events if e["type"] == "content_block_start"][0]
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "get_weather"
    deltas = [e["delta"] for _, e in events if e["type"] == "content_block_delta"]
    expected = {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'}
    assert deltas[-1] == expected
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


async def test_gemini_thinking_then_text(gemini_proxy):
    backend, router = gemini_proxy
    backend.stream_chunks = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "reasoning", "thought": True},
                            {"text": "answer"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
        {"usageMetadata": GEMINI_USAGE},
    ]
    proto = await post(router, "/v1/messages", _messages_body(stream=True))
    events = _parse_sse(proto)
    block_starts = [
        e["content_block"]["type"]
        for _, e in events
        if e["type"] == "content_block_start"
    ]
    assert block_starts == ["thinking", "text"]


async def test_gemini_tool_turn_round_trips(gemini_proxy):
    """An assistant tool_use + tool_result round-trips through Gemini's
    functionCall/functionResponse wire shapes."""
    backend, router = gemini_proxy
    body = _messages_body(
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": '{"temp": 18}',
                    }
                ],
            },
        ]
    )
    await post(router, "/v1/messages", body)
    contents = backend.last_body["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"city": "Paris"},
                    "id": "call_1",
                }
            }
        ],
    }
    assert contents[2]["parts"][0]["functionResponse"] == {
        "name": "call_1",
        "response": {"temp": 18},
    }
