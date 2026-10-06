from __future__ import annotations

import asyncio
import hashlib
import os
import re
import socket
import time
from http import HTTPStatus
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec
from granian._granian import RSGIHTTPProtocol, RSGIProtocolClosed
from gravier import AddressInUseError, Response, Router, RSGIScope, serve
from padwan_ai._json import dumps as _json_dumps, loads as _json_loads
from padwan_ai.anthropic.compat import _image_source_to_part, messages_to_openai
from padwan_ai.anthropic.events import (
    error_to_anthropic,
    response_to_anthropic,
    stream_to_anthropic,
)
from padwan_ai.anthropic.models import AnthropicCompatBody
from padwan_ai.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_ai.openai.client import OpenAIClient, _OpenAIBase
from piou import CommandError, Option

from .breakdown import format_breakdown, format_tree, prompt_breakdown
from .claude_config import client_base_url, write_claude_config
from .defaults import (
    DEFAULT_APPROVAL_CONFIDENCE,
    DEFAULT_LAYA_MODEL,
    DEFAULT_STREAM_RETRIES,
    DEFAULT_TIMEOUT,
    ENV_PREFIX,
    PADWAN_API_KEY_ENV,
    PADWAN_BASE_URL_ENV,
)
from .logs import log, log_request, route, timing_detail
from .systemone import Laya, SystemOneRequest
from .trace import session_context
from .utils import console, startup_banner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from padwan_ai.anthropic.models import CountTokensResponse
    from padwan_ai.openai.types import CreateChatCompletionRequest
    from padwan_ai.typesafe.models import Question

__all__ = ("build_router", "main", "proxy_command")

SSE_HEADERS = [
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache, no-transform"),
    ("x-accel-buffering", "no"),
]

_RETRY_BACKOFF = 0.5

_CONNECT_TIMEOUT = 10.0


def _port_answers(host: str, port: int) -> bool:
    """Whether something already serves this port.

    Granian binds with SO_REUSEPORT, so a second proxy silently shares the port
    instead of failing, and requests round-robin between two different configs.
    """
    with socket.socket() as probe:
        probe.settimeout(0.5)
        dialable = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return probe.connect_ex((dialable, port)) == 0


_MAX_SYSTEM_ONE_BODY = 64 * 1024

_CLIENT_STATUS = re.compile(r"^\[[\w-]+\] (4\d\d)")


def _retryable_stream_error(e: Exception) -> bool:
    """Whether a failed stream attempt could succeed if replayed.

    Rate limits carry their own retry-after, quota and 4xx are permanent;
    timeouts, connection resets and 5xx are worth another attempt.
    """
    match e:
        case TooManyRequestsError() | QuotaExceededError():
            return False
        case LLMError():
            return _CLIENT_STATUS.match(str(e)) is None
        case _:
            return True


def _error_detail(e: Exception) -> str:
    """Describe an exception even when it has no message."""
    detail = str(e) or repr(e)
    cause: BaseException | None = e
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        response = getattr(cause, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return f"HTTP {status}: {detail}"
        cause = cause.__cause__ or cause.__context__
    return detail


async def _timed_chunks(
    chunks: AsyncIterator[Any], wait: list[float]
) -> AsyncIterator[Any]:
    """Pass chunks through, accumulating time spent awaiting the backend in wait[0]."""
    while True:
        t0 = time.perf_counter()
        try:
            chunk = await anext(chunks)
        except StopAsyncIteration:
            wait[0] += time.perf_counter() - t0
            return
        wait[0] += time.perf_counter() - t0
        yield chunk


def _normalize_reasoning(data: dict[str, Any]) -> None:
    """Expose Scaleway's reasoning alias to the Anthropic translator."""
    for choice in data.get("choices") or []:
        for key in ("message", "delta"):
            message = choice.get(key)
            if isinstance(message, dict) and "reasoning_content" not in message:
                reasoning = message.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    message["reasoning_content"] = reasoning


def _validate_body(body: AnthropicCompatBody) -> None:
    """Validate Claude Code's inline system extension without changing the request."""
    projection = body
    if (
        isinstance(body, dict)
        and isinstance(body.get("messages"), list)
        and any(
            isinstance(m, dict) and m.get("role") == "system" for m in body["messages"]
        )
    ):
        messages = []
        for message in body["messages"]:
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                if not isinstance(content, str) and not (
                    isinstance(content, list)
                    and all(
                        isinstance(block, dict) and block.get("type") == "text"
                        for block in content
                    )
                ):
                    raise ValueError("inline system messages require text content")
                message = {**message, "role": "user"}
            messages.append(message)
        projection = {**body, "messages": messages}
    msgspec.convert(projection, type=AnthropicCompatBody)


def _translate_body(
    body: AnthropicCompatBody, *, model: str
) -> CreateChatCompletionRequest:
    """Preserve inline system roles and ordering through the upstream translator."""
    translated = messages_to_openai(body, model=model)
    if not any(message["role"] == "system" for message in body["messages"]):
        return translated
    # Keep full-conversation tool selection; rebuild only the message sequence.
    # Segments carry no tools: translating messages needs none, and tools cost O(n).
    base = cast(AnthropicCompatBody, {k: v for k, v in body.items() if k != "tools"})
    translated["messages"] = messages_to_openai({**base, "messages": []})["messages"]
    for message in body["messages"]:
        segment: AnthropicCompatBody = {**base, "system": "", "messages": [message]}
        if message["role"] == "system":
            segment = {**base, "system": message["content"], "messages": []}
        translated["messages"].extend(messages_to_openai(segment)["messages"])
    return translated


async def _prepare_chunks(
    chunks: AsyncIterator[Any], seen: list[bool]
) -> AsyncIterator[Any]:
    """Normalize reasoning fields and mark the first backend chunk."""
    async for chunk in chunks:
        _normalize_reasoning(chunk)
        seen[0] = True
        yield chunk


def _make_client(
    backend_url: str | None,
    model: str,
    api_key_env: str | None,
    timeout: float = DEFAULT_TIMEOUT,
) -> _OpenAIBase:
    """Build the backend client with explicit, Padwan, then OpenAI key precedence."""
    backend_url = backend_url or os.environ.get(PADWAN_BASE_URL_ENV)
    if not backend_url:
        raise CommandError(
            f"No backend URL: pass --backend-url or set {PADWAN_BASE_URL_ENV}"
        )
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise CommandError(f"{api_key_env} not set")
    else:
        api_key = os.environ.get(PADWAN_API_KEY_ENV) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[yellow]No backend API key set; connecting to the backend "
            "unauthenticated (fine for local servers)[/yellow]"
        )
    client_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": backend_url,
        "api_key": api_key or "no-key-required",
        # (connect, read) tuple: a read timeout long enough for a silent
        # reasoning model must not also let a dead host hang that long.
        "timeout": cast(float, (_CONNECT_TIMEOUT, timeout)),
    }
    return OpenAIClient(**client_kwargs)


def _pick_model(requested: str, *, model: str, small_model: str | None) -> str:
    """Route the requested Anthropic model to a backend model.

    Anthropic clients use their small model tier (haiku) for lightweight
    internal calls; everything else gets the main model. A client configured
    with the backend names (`--claude-config`) already asks for them directly.
    """
    if requested in (model, small_model):
        return requested
    if small_model and "haiku" in requested:
        return small_model
    return model


def _has_images(messages: list[Any]) -> bool:
    """True if any message carries an image block, including inside tool results."""
    return bool(_image_blocks(messages))


_CAPTION_PROMPT = (
    "Describe this image for an assistant that cannot see it. Transcribe all "
    "visible text verbatim, then describe the layout, colors, UI elements, charts "
    "and anything else notable. If there is no text, say so; never invent content."
)
_CAPTION_MAX_TOKENS = 1024
# ponytail: FIFO eviction, per worker; LRU if long sessions thrash it
_CAPTION_CACHE_SIZE = 256


def _image_blocks(messages: list[Any]) -> list[tuple[list[Any], int]]:
    """Locate image blocks as (list, index), including inside tool results."""
    found: list[tuple[list[Any], int]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                found.append((content, i))
            elif block.get("type") == "tool_result" and isinstance(
                inner := block.get("content"), list
            ):
                found.extend(
                    (inner, j)
                    for j, b in enumerate(inner)
                    if isinstance(b, dict) and b.get("type") == "image"
                )
    return found


async def _caption_images(
    messages: list[Any], *, client: _OpenAIBase, model: str, cache: dict[str, str]
) -> int:
    """Replace image blocks in place with vision-model captions; return how many."""
    blocks = _image_blocks(messages)
    keys = [
        hashlib.sha256(_json_dumps(c[i].get("source") or {}).encode()).hexdigest()
        for c, i in blocks
    ]
    todo = {k: c[i] for k, (c, i) in zip(keys, blocks) if k not in cache}

    async def caption(block: dict[str, Any]) -> str:
        part = _image_source_to_part(block.get("source") or {})
        if part is None:
            return "unsupported image source"
        body = {
            "model": model,
            "max_tokens": _CAPTION_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _CAPTION_PROMPT}, part],
                }
            ],
        }
        data, _ = await client.complete(cast(Any, body))
        return cast(Any, data)["choices"][0]["message"].get("content") or ""

    # Request-local copy: eviction below must not drop a caption this request needs.
    texts = {k: cache[k] for k in keys if k in cache}
    for key, text in zip(todo, await asyncio.gather(*map(caption, todo.values()))):
        if len(cache) >= _CAPTION_CACHE_SIZE:
            del cache[next(iter(cache))]
        cache[key] = texts[key] = text
    for key, (container, i) in zip(keys, blocks):
        container[i] = {"type": "text", "text": f"[Image description]\n{texts[key]}"}
    return len(blocks)


_SESSION_RE = re.compile(r"session_([0-9a-fA-F-]{36})")


def _client_session(body: dict[str, Any]) -> str | None:
    """Claude Code's session id from `metadata.user_id` (JSON or `…_session_<uuid>`)."""
    user_id = (body.get("metadata") or {}).get("user_id")
    if not isinstance(user_id, str):
        return None
    if user_id.startswith("{"):
        try:
            parsed = _json_loads(user_id)
        except ValueError:
            return None
        session = parsed.get("session_id") if isinstance(parsed, dict) else None
        return session if isinstance(session, str) and session else None
    match = _SESSION_RE.search(user_id)
    return match[1] if match else None


def _sse(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {_json_dumps(payload)}\n\n"


def _tool_names(content: list[Any]) -> list[str]:
    """Names of the tool_use blocks in an Anthropic response body."""
    return [
        block.get("name") or "?"
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def build_router(
    *,
    client: _OpenAIBase,
    model: str,
    small_model: str | None = None,
    vision_model: str | None = None,
    vision_mode: str = "route",
    max_output_tokens: int = 16384,
    stream_retries: int = DEFAULT_STREAM_RETRIES,
    timings: bool = False,
    breakdown: bool = False,
    rich: bool = False,
    laya: Laya | None = None,
) -> Router:
    """Build the Anthropic-compatible RSGI router over an OpenAI-compatible client."""
    router = Router()
    captions: dict[str, str] = {}

    async def _stream(
        proto: RSGIHTTPProtocol,
        request_body: dict[str, Any],
        requested_model: str,
        target_model: str,
        req_xlate: float,
        detail: str,
        session: str | None,
    ) -> None:
        # Client disconnect mid-stream closes the transport; expected, not a fault.
        try:
            await _stream_body(
                proto,
                request_body,
                requested_model,
                target_model,
                req_xlate,
                detail,
                session,
            )
        except RSGIProtocolClosed:
            log.info("%s | client disconnected", route(requested_model, target_model))

    async def _stream_body(
        proto: RSGIHTTPProtocol,
        request_body: dict[str, Any],
        requested_model: str,
        target_model: str,
        req_xlate: float,
        detail: str,
        session: str | None,
    ) -> None:
        start = time.monotonic()
        backend_wait = [0.0]
        # Without this OpenAI-compatible backends omit usage from the final chunk.
        request_body.setdefault("stream_options", {"include_usage": True})
        transport = proto.response_stream(200, SSE_HEADERS)
        sent = False
        attempt = 0
        while True:
            attempt_start = time.monotonic()
            usage: dict[str, Any] = {}
            stop_reason: str | None = None
            tools_called: list[str] = []
            first_chunk = [False]
            try:
                chunks: AsyncIterator[Any] = client.stream(cast(Any, request_body))
                if timings:
                    chunks = _timed_chunks(chunks, backend_wait)
                prepared = _prepare_chunks(chunks, first_chunk)
                events = stream_to_anthropic(prepared, model=requested_model)
                # message_start is emitted before any backend I/O; hold frames
                # until the backend produced a chunk so a failed attempt stays
                # un-sent — and therefore replayable.
                held: list[str] = []
                async for name, payload in events:
                    if name == "message_delta":
                        usage = payload.get("usage") or {}
                        stop_reason = (payload.get("delta") or {}).get("stop_reason")
                    elif name == "content_block_start":
                        block = payload.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            tools_called.append(block.get("name") or "?")
                    if not first_chunk[0]:
                        held.append(_sse(name, payload))
                        continue
                    sent = True
                    for frame in held:
                        await transport.send_str(frame)
                    held.clear()
                    await transport.send_str(_sse(name, payload))
                for frame in held:  # backend closed without emitting any chunk
                    await transport.send_str(frame)
                break
            except RSGIProtocolClosed:  # client gone: don't retry, don't report in-band
                raise
            except Exception as e:  # error mid-stream: report in-band, Anthropic style
                if sent or attempt >= stream_retries or not _retryable_stream_error(e):
                    log.warning(
                        "%s | stream failed after %.2fs: %s",
                        f"{session[:8]} {route(requested_model, target_model)}"
                        if session
                        else route(requested_model, target_model),
                        time.monotonic() - start,
                        _error_detail(e),
                    )
                    _, body = error_to_anthropic(e)
                    await transport.send_str(_sse("error", body))
                    return
                attempt += 1
                log.warning(
                    "%s | stream attempt %d failed after %.2fs: %s — retrying",
                    f"{session[:8]} {route(requested_model, target_model)}"
                    if session
                    else route(requested_model, target_model),
                    attempt,
                    time.monotonic() - attempt_start,
                    _error_detail(e),
                )
                await asyncio.sleep(_RETRY_BACKOFF * 2 ** (attempt - 1))
        elapsed = time.monotonic() - start
        log_request(
            requested_model,
            target_model,
            kind="stream",
            usage=usage,
            stop_reason=stop_reason,
            elapsed=elapsed,
            tools=tools_called,
            timing=timing_detail(elapsed + req_xlate, backend_wait[0], req_xlate)
            if timings
            else None,
            breakdown=detail,
            session=session,
        )

    @router.post("/v1/messages")
    async def messages(scope: RSGIScope, proto: RSGIHTTPProtocol) -> Response | None:
        try:
            body = cast("AnthropicCompatBody", _json_loads(await proto()))
            # Validate known fields while retaining extension fields for translation.
            _validate_body(body)
            if not body["model"] or not body["messages"] or body["max_tokens"] <= 0:
                raise ValueError(
                    "model and messages must be nonempty; max_tokens must be positive"
                )
            requested_model = body["model"]
            target = _pick_model(requested_model, model=model, small_model=small_model)
            # Image-bearing requests need a multimodal backend, whatever the tier.
            if vision_model and _has_images(cast(list, body["messages"])):
                if vision_mode == "route":
                    target = vision_model
                else:
                    captioned = await _caption_images(
                        cast(list, body["messages"]),
                        client=client,
                        model=vision_model,
                        cache=captions,
                    )
                    log.info("captioned %d image(s) with %s", captioned, vision_model)
            # Anthropic clients ask for large budgets (32k); backends cap lower.
            body["max_tokens"] = min(body["max_tokens"], max_output_tokens)
            xlate_start = time.perf_counter()
            backend_body = cast("dict[str, Any]", _translate_body(body, model=target))
            req_xlate = time.perf_counter() - xlate_start
            fmt = format_tree if rich else format_breakdown
            detail = fmt(prompt_breakdown(body)) if breakdown else ""
            session = _client_session(cast("dict[str, Any]", body))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            return Response(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": str(e)},
                },
                status=HTTPStatus.BAD_REQUEST,
            )
        if body.get("stream"):
            with session_context(session):
                await _stream(
                    proto,
                    cast("dict[str, Any]", backend_body),
                    requested_model,
                    target,
                    req_xlate,
                    detail,
                    session,
                )
            return None
        start = time.monotonic()
        try:
            with session_context(session):
                data, _ = await client.complete(cast(Any, backend_body))
        except Exception as e:
            log.warning(
                "%s | request failed after %.2fs: %s",
                f"{session[:8]} {route(requested_model, target)}"
                if session
                else route(requested_model, target),
                time.monotonic() - start,
                _error_detail(e),
            )
            status, error_body = error_to_anthropic(e)
            return Response(error_body, status=status)
        backend_s = time.monotonic() - start
        _normalize_reasoning(cast("dict[str, Any]", data))
        resp = response_to_anthropic(cast(Any, data), model=requested_model)
        elapsed = time.monotonic() - start + req_xlate
        log_request(
            requested_model,
            target,
            kind="complete",
            usage=cast("dict[str, Any]", resp.get("usage") or {}),
            stop_reason=resp.get("stop_reason"),
            elapsed=elapsed,
            tools=_tool_names(cast("list[Any]", resp.get("content") or [])),
            timing=timing_detail(elapsed, backend_s, req_xlate) if timings else None,
            breakdown=detail,
            session=session,
        )
        return Response(resp)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(scope: RSGIScope, proto: RSGIHTTPProtocol) -> Response:
        # chars/4 heuristic straight off the raw body: no parse round-trip
        estimate: CountTokensResponse = {
            "input_tokens": max(1, len(await proto()) // 4)
        }
        return Response(estimate)

    if laya is not None:
        # TypeSafe's System One route, answered locally: --approvals laya points
        # the Claude Code hook here instead of api.typesafe.ai.
        @router.post("/systemone")
        async def system_one(scope: RSGIScope, proto: RSGIHTTPProtocol) -> Response:
            raw = await proto()
            if len(raw) > _MAX_SYSTEM_ONE_BODY:
                return Response(
                    {"error": "request too large"},
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            try:
                body = msgspec.convert(_json_loads(raw), type=SystemOneRequest)
            except (msgspec.ValidationError, ValueError) as e:
                return Response({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            if not body["questions"]:
                return Response(
                    {"error": "questions must not be empty"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            for question in body["questions"].values():
                kind, criteria = question.get("type"), question.get("criteria")
                if (
                    not isinstance(question.get("instructions"), str)
                    or not isinstance(kind, str)
                    or kind not in {"choice", "score", "noul"}
                    or (
                        kind in {"choice", "score"}
                        and (
                            not isinstance(criteria, (dict, list))
                            or not 2 <= len(criteria) <= 20
                            or (kind == "score" and not isinstance(criteria, list))
                            or (
                                kind == "choice"
                                and isinstance(criteria, list)
                                and not all(isinstance(c, str) for c in criteria)
                            )
                        )
                    )
                    or (
                        kind == "noul"
                        and criteria is not None
                        and (
                            not isinstance(criteria, dict)
                            or not set(criteria) <= {"true", "false"}
                        )
                    )
                ):
                    return Response(
                        {"error": "invalid question schema or option count (2–20)"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
            # Laya silently drops the state tail past its window, which would hide
            # the very tool call being judged: refuse instead of answering blind.
            if not laya.fits(body["state"]):
                log.warning(
                    "systemone | state over the %d-token window, refused",
                    laya.state_budget,
                )
                return Response(
                    {"error": f"state exceeds {laya.state_budget} tokens"},
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            start = time.monotonic()
            try:
                answers = await laya.system_one(
                    body["state"], cast("dict[str, Question]", body["questions"])
                )
            except Exception as e:
                log.warning("systemone | local inference failed: %s", e)
                return Response(
                    {"error": "local inference failed"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            log.info(
                "systemone | %d question(s) in %.0fms",
                len(body["questions"]),
                (time.monotonic() - start) * 1000,
            )
            return Response(answers)

    return router


def proxy_command(
    backend_url: str | None = Option(
        None,
        "--backend-url",
        help="OpenAI-compatible endpoint (default: $PADWAN_BASE_URL)",
    ),
    model: str = Option(..., "-m", "--model", help="Backend model for main requests"),
    small_model: str | None = Option(
        None, "--small-model", help="Backend model for haiku-tier requests"
    ),
    vision_model: str | None = Option(
        None,
        "--vision-model",
        help="Multimodal backend model for requests carrying images",
    ),
    vision_mode: str = Option(
        "route",
        "--vision-mode",
        choices=["route", "caption"],
        help="`route` sends image requests to --vision-model; `caption` has it "
        "describe each image as text for the main model (for vision models "
        "without tool calling, e.g. pixtral)",
    ),
    api_key_env: str | None = Option(
        None,
        "--api-key-env",
        help="Env var holding the backend API key "
        "(default: PADWAN_API_KEY, then OPENAI_API_KEY)",
    ),
    max_output_tokens: int = Option(
        16384, "--max-output-tokens", help="Cap on max_tokens forwarded to the backend"
    ),
    timeout: float = Option(
        DEFAULT_TIMEOUT,
        "--timeout",
        help="Backend read timeout in seconds, applied per stream gap "
        "(reasoning models can stay silent for minutes)",
    ),
    claude_config: Path | None = Option(
        None,
        "--claude-config",
        help="Write Claude Code settings into this config directory",
        raise_path_does_not_exist=False,
    ),
    context_window: int | None = Option(
        None,
        "--context-window",
        help="Backend context window in tokens for Claude Code auto-compaction",
    ),
    approvals: str | None = Option(
        None,
        "--approvals",
        choices=["jev", "laya"],
        help="Tool approval backend: `jev` calls TypeSafe; `laya` provides local "
        "advisory verdicts and always asks (needs the `laya` extra)",
    ),
    no_approvals: bool = Option(
        False,
        "--no-approvals",
        help="Remove the approval hook from the Claude Code config",
    ),
    approval_model: str | None = Option(
        None,
        "--approval-model",
        help="Checkpoint or API model to approve with "
        "(default: jev-latest, or convaiinnovations/laya)",
    ),
    approval_confidence: float | None = Option(
        None,
        "--approval-confidence",
        help="Hosted approval confidence gate (default 0.99); Laya stays advisory",
    ),
    approval_subfolder: str | None = Option(
        None,
        "--approval-subfolder",
        help="Laya checkpoint subfolder, e.g. multilingual or typed-decisions",
    ),
    stream_retries: int = Option(
        DEFAULT_STREAM_RETRIES,
        "--stream-retries",
        help="Replays of a stream that fails before any event reaches the client",
    ),
    host: str = Option("127.0.0.1", "--host", help="Bind address"),
    port: int = Option(4000, "-p", "--port", help="Port to listen on"),
    trace: bool = Option(
        False, "--trace", help="Instrument proxied requests (Langfuse or OTLP export)"
    ),
    trace_content: bool = Option(
        False,
        "--trace-content",
        help="Like --trace, plus prompts and completions recorded on the spans",
    ),
    verbose: bool = Option(
        False,
        "-v",
        "--verbose",
        help="Log each proxied request (models, tokens, duration)",
    ),
    breakdown: bool = Option(
        False,
        "--breakdown",
        help="Like -v, plus a per-request prompt split: system, tool schemas "
        "(grouped by MCP server), and message history",
    ),
    rich: bool = Option(
        False,
        "--rich",
        help="Render the -v request log with colours and aligned columns",
    ),
    timings: bool = Option(
        False,
        "-vv",
        "--timings",
        help="Like -v, plus per-request timing split: backend wait, "
        "request-translation time, proxy overhead",
    ),
) -> None:
    """Serve the Anthropic Messages API over an OpenAI-compatible backend.

    Point an Anthropic client at it, e.g.:
    ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
    """
    if context_window is not None and context_window <= 0:
        raise CommandError("--context-window must be greater than zero")
    if approvals and no_approvals:
        raise CommandError("pass only one of --approvals or --no-approvals")
    # Jev lives entirely in the hook, so it is useless without settings to write;
    # laya also serves /systemone, which is the whole point of the Docker image.
    if (approvals == "jev" or no_approvals) and claude_config is None:
        flag = "--approvals jev" if approvals else "--no-approvals"
        raise CommandError(f"{flag} requires --claude-config")
    if approvals == "laya" and find_spec("laya") is None:
        raise CommandError(
            "--approvals laya needs the `laya` extra: uv sync --extra laya"
        )
    if approval_model and not approvals:
        raise CommandError("--approval-model requires --approvals")
    if approval_subfolder and approvals != "laya":
        raise CommandError("--approval-subfolder requires --approvals laya")
    if approval_confidence is not None:
        if not approvals:
            raise CommandError("--approval-confidence requires --approvals")
        if not 0 < approval_confidence <= 1:
            raise CommandError("--approval-confidence must be within (0, 1]")

    if _port_answers(host, port):
        raise CommandError(
            f"port {port} is already served; stop that proxy first "
            "(granian would share the port instead of failing)"
        )

    # Validate configuration in the CLI process for clean errors; workers
    # rebuild the client from the env snapshot below.
    _make_client(backend_url, model, api_key_env)

    resolved_backend_url = backend_url or os.environ.get(PADWAN_BASE_URL_ENV)
    local = approvals == "laya"
    approval_model = approval_model or (DEFAULT_LAYA_MODEL if local else None)
    approvals_config = True if approvals else False if no_approvals else None
    if claude_config is not None:
        write_claude_config(
            claude_config,
            host=host,
            port=port,
            model=model,
            small_model=small_model,
            backend_url=resolved_backend_url,
            timeout=timeout,
            context_window=context_window,
            approvals=approvals_config,
            approvals_url=client_base_url(host, port) if local else None,
            # The proxy already loaded the local checkpoint; the hook needs the
            # name only to pick a hosted model.
            approvals_model=None if local else approval_model,
            approvals_confidence=approval_confidence,
        )

    env: dict[str, str | None] = {
        "BACKEND_URL": backend_url,
        "MODEL": model,
        "SMALL_MODEL": small_model,
        "VISION_MODEL": vision_model,
        "VISION_MODE": vision_mode,
        "API_KEY_ENV": api_key_env,
        "MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "TIMEOUT": str(timeout),
        "STREAM_RETRIES": str(stream_retries),
        "TRACE": "1" if (trace or trace_content) else "",
        "TRACE_CONTENT": "1" if trace_content else "",
        "VERBOSE": "1" if (verbose or timings or breakdown) else "",
        "TIMINGS": "1" if timings else "",
        "BREAKDOWN": "1" if breakdown else "",
        "RICH": "1" if rich else "",
        "LAYA_MODEL": approval_model if local else "",
        "LAYA_SUBFOLDER": approval_subfolder if local else "",
    }
    for key, value in env.items():
        if value:
            os.environ[ENV_PREFIX + key] = value
        else:
            os.environ.pop(ENV_PREFIX + key, None)

    console.print(
        startup_banner(
            {
                "listen": f"http://{host}:{port}",
                "backend": resolved_backend_url or "?",
                "model": model,
                "small": small_model or f"{model} [dim](same)[/dim]",
                "vision": f"{vision_model} [dim]({vision_mode})[/dim]"
                if vision_model
                else "[dim]none[/dim]",
                "approvals": {
                    True: f"[green]{approval_model or approvals}[/green]"
                    + (
                        " [dim]local advisory (always ask)[/dim]"
                        if local
                        else " [dim]over "
                        f"{approval_confidence or DEFAULT_APPROVAL_CONFIDENCE}[/dim]"
                    ),
                    False: "[dim]hook disabled[/dim]",
                    None: "[dim]hook unchanged[/dim]",
                }[approvals_config],
            }
        )
    )
    try:
        serve("padwan_proxy.rsgi:app", host=host, port=port)
    except AddressInUseError as e:
        raise CommandError(str(e)) from e


def main() -> None:
    """Entry point for the `padwan-proxy` script."""
    from piou import Cli

    cli = Cli(description="Anthropic Messages API over any OpenAI-compatible backend")
    cli.main(help="Serve the proxy")(proxy_command)
    cli.run()
