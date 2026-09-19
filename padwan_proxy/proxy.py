from __future__ import annotations

import asyncio
import os
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec
from granian._granian import RSGIHTTPProtocol, RSGIProtocolClosed
from gravier import AddressInUseError, Response, Router, RSGIScope, serve
from padwan_llm._json import dumps as _json_dumps, loads as _json_loads
from padwan_llm.anthropic.compat import messages_to_openai
from padwan_llm.anthropic.events import (
    error_to_anthropic,
    response_to_anthropic,
    stream_to_anthropic,
)
from padwan_llm.anthropic.models import AnthropicCompatBody
from padwan_llm.client import PADWAN_API_KEY_ENV, PADWAN_BASE_URL_ENV
from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_llm.openai.client import OpenAIClient, _OpenAIBase
from piou import CommandError, Option

from .breakdown import format_breakdown, prompt_breakdown
from .claude_config import write_claude_config
from .defaults import DEFAULT_STREAM_RETRIES, DEFAULT_TIMEOUT, ENV_PREFIX
from .logs import log, log_request, route, timing_detail
from .trace import session_context
from .utils import console

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from padwan_llm.anthropic.models import CountTokensResponse
    from padwan_llm.openai.types import CreateChatCompletionRequest

__all__ = ("build_router", "main", "proxy_command")

SSE_HEADERS = [
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache, no-transform"),
    ("x-accel-buffering", "no"),
]

_RETRY_BACKOFF = 0.5

_CONNECT_TIMEOUT = 10.0

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
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
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
    translated["messages"] = messages_to_openai({**body, "messages": []})["messages"]
    for message in body["messages"]:
        segment: AnthropicCompatBody = {**body, "system": "", "messages": [message]}
        if message["role"] == "system":
            segment = {**body, "system": message["content"], "messages": []}
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
    return OpenAIClient(
        model=model,
        base_url=backend_url,
        api_key=api_key or "no-key-required",
        # (connect, read) tuple: a read timeout long enough for a silent
        # reasoning model must not also let a dead host hang that long.
        timeout=cast(float, (_CONNECT_TIMEOUT, timeout)),
    )


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
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                return True
            if block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list) and any(
                    isinstance(b, dict) and b.get("type") == "image" for b in inner
                ):
                    return True
    return False


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


def build_router(
    *,
    client: _OpenAIBase,
    model: str,
    small_model: str | None = None,
    vision_model: str | None = None,
    max_output_tokens: int = 16384,
    stream_retries: int = DEFAULT_STREAM_RETRIES,
    timings: bool = False,
    breakdown: bool = False,
) -> Router:
    """Build the Anthropic-compatible RSGI router over an OpenAI-compatible client."""
    router = Router()

    async def _stream(
        proto: RSGIHTTPProtocol,
        request_body: dict[str, Any],
        requested_model: str,
        target_model: str,
        req_xlate: float,
        detail: str,
    ) -> None:
        # Client disconnect mid-stream closes the transport; expected, not a fault.
        try:
            await _stream_body(
                proto, request_body, requested_model, target_model, req_xlate, detail
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
            first_chunk = [False]
            try:
                chunks: AsyncIterator[Any] = client.stream(cast(Any, request_body))
                if timings:
                    chunks = _timed_chunks(chunks, backend_wait)
                events = stream_to_anthropic(
                    _prepare_chunks(chunks, first_chunk), model=requested_model
                )
                # message_start is emitted before any backend I/O; hold frames
                # until the backend produced a chunk so a failed attempt stays
                # un-sent — and therefore replayable.
                held: list[str] = []
                async for name, payload in events:
                    if name == "message_delta":
                        usage = payload.get("usage") or {}
                        stop_reason = (payload.get("delta") or {}).get("stop_reason")
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
                        route(requested_model, target_model),
                        time.monotonic() - start,
                        e,
                    )
                    _, body = error_to_anthropic(e)
                    await transport.send_str(_sse("error", body))
                    return
                attempt += 1
                log.warning(
                    "%s | stream attempt %d failed after %.2fs: %s — retrying",
                    route(requested_model, target_model),
                    attempt,
                    time.monotonic() - attempt_start,
                    e,
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
            timing=timing_detail(elapsed + req_xlate, backend_wait[0], req_xlate)
            if timings
            else "",
            breakdown=detail,
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
                target = vision_model
            # Anthropic clients ask for large budgets (32k); backends cap lower.
            body["max_tokens"] = min(body["max_tokens"], max_output_tokens)
            xlate_start = time.perf_counter()
            openai_body = _translate_body(body, model=target)
            req_xlate = time.perf_counter() - xlate_start
            detail = format_breakdown(prompt_breakdown(body)) if breakdown else ""
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
                    cast("dict[str, Any]", openai_body),
                    requested_model,
                    target,
                    req_xlate,
                    detail,
                )
            return None
        start = time.monotonic()
        try:
            with session_context(session):
                data, _ = await client.complete(openai_body)
        except Exception as e:
            log.warning(
                "%s | request failed after %.2fs: %s",
                route(requested_model, target),
                time.monotonic() - start,
                e,
            )
            status, error_body = error_to_anthropic(e)
            return Response(error_body, status=status)
        backend_s = time.monotonic() - start
        _normalize_reasoning(cast("dict[str, Any]", data))
        resp = response_to_anthropic(data, model=requested_model)
        elapsed = time.monotonic() - start + req_xlate
        log_request(
            requested_model,
            target,
            kind="complete",
            usage=cast("dict[str, Any]", resp.get("usage") or {}),
            stop_reason=resp.get("stop_reason"),
            elapsed=elapsed,
            timing=timing_detail(elapsed, backend_s, req_xlate) if timings else "",
            breakdown=detail,
        )
        return Response(resp)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(scope: RSGIScope, proto: RSGIHTTPProtocol) -> Response:
        # chars/4 heuristic straight off the raw body: no parse round-trip
        estimate: CountTokensResponse = {
            "input_tokens": max(1, len(await proto()) // 4)
        }
        return Response(estimate)

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

    # Validate configuration in the CLI process for clean errors; workers
    # rebuild the client from the env snapshot below.
    _make_client(backend_url, model, api_key_env)

    resolved_backend_url = backend_url or os.environ.get(PADWAN_BASE_URL_ENV)
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
        )

    env: dict[str, str | None] = {
        "BACKEND_URL": backend_url,
        "MODEL": model,
        "SMALL_MODEL": small_model,
        "VISION_MODEL": vision_model,
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
    }
    for key, value in env.items():
        if value:
            os.environ[ENV_PREFIX + key] = value
        else:
            os.environ.pop(ENV_PREFIX + key, None)

    console.print(
        f"[green]Anthropic-compatible proxy on http://{host}:{port} "
        f"→ {resolved_backend_url} "
        f"(model={model}, small={small_model or model}, "
        f"vision={vision_model or 'none'})[/green]"
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
