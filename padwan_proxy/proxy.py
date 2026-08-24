from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, cast

from granian._granian import RSGIHTTPProtocol
from gravier import AddressInUseError, Response, Router, RSGIScope, serve
from padwan_llm._json import dumps as _json_dumps, loads as _json_loads
from padwan_llm.anthropic.compat import messages_to_openai
from padwan_llm.anthropic.events import (
    error_to_anthropic,
    response_to_anthropic,
    stream_to_anthropic,
)
from padwan_llm.client import PADWAN_API_KEY_ENV, PADWAN_BASE_URL_ENV, LLMClient
from padwan_llm.openai.client import _OpenAIBase
from piou import CommandError, Option

from .utils import console

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from padwan_llm.anthropic.models import CountTokensResponse, MessagesBody

__all__ = ("build_router", "main", "proxy_command")

log = logging.getLogger("padwan_proxy.proxy")

SSE_HEADERS = [
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache, no-transform"),
    ("x-accel-buffering", "no"),
]

# env round-trip between the CLI process and granian worker processes
ENV_PREFIX = "PADWAN_PROXY_"


def _log_request(
    requested: str,
    target: str,
    *,
    kind: str,
    usage: dict[str, Any],
    stop_reason: str | None,
    elapsed: float,
    timing: str = "",
) -> None:
    cached = usage.get("cache_read_input_tokens")
    log.info(
        "%s → %s | %s | %s | in=%s out=%s%s | %.2fs%s",
        requested,
        target,
        kind,
        stop_reason or "?",
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        f" cached={cached}" if cached else "",
        elapsed,
        timing,
    )


def _timing_detail(elapsed: float, backend: float, req_xlate: float) -> str:
    """Format the -vv timing segment: backend wait vs time spent in the proxy."""
    overhead = max(0.0, elapsed - backend - req_xlate)
    return (
        f" (backend {backend:.2f}s, req-xlate {req_xlate * 1e3:.1f}ms, "
        f"proxy {overhead * 1e3:.1f}ms)"
    )


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


def _make_client(
    backend_url: str | None, model: str, api_key_env: str | None
) -> _OpenAIBase:
    """Build the backend client, delegating env resolution to `LLMClient`.

    padwan-llm handles `PADWAN_BASE_URL` gateway mode and prefers
    `PADWAN_API_KEY` for custom endpoints; an explicit ``--api-key-env``
    overrides both. Without any URL, `LLMClient` would route by model name
    to a provider's native endpoint — never what a proxy wants, so refuse.
    """
    if not (backend_url or os.environ.get(PADWAN_BASE_URL_ENV)):
        raise CommandError(
            f"No backend URL: pass --backend-url or set {PADWAN_BASE_URL_ENV}"
        )
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise CommandError(f"{api_key_env} not set")
    if api_key is None and not os.environ.get(PADWAN_API_KEY_ENV):
        console.print(
            f"[yellow]{PADWAN_API_KEY_ENV} not set; connecting to the backend "
            "unauthenticated (fine for local servers)[/yellow]"
        )
    client = LLMClient(model=model, base_url=backend_url, api_key=api_key)
    if not isinstance(client, _OpenAIBase):
        raise CommandError(f"Backend for {model!r} is not OpenAI-compatible")
    return client


def _pick_model(requested: str, *, model: str, small_model: str | None) -> str:
    """Route the requested Anthropic model to a backend model.

    Anthropic clients use their small model tier (haiku) for lightweight
    internal calls; everything else gets the main model.
    """
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


def _sse(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {_json_dumps(payload)}\n\n"


def build_router(
    *,
    client: _OpenAIBase,
    model: str,
    small_model: str | None = None,
    vision_model: str | None = None,
    max_output_tokens: int = 16384,
    timings: bool = False,
) -> Router:
    """Build the Anthropic-compatible RSGI router over an OpenAI-compatible client."""
    router = Router()

    async def _stream(
        proto: RSGIHTTPProtocol,
        request_body: dict[str, Any],
        requested_model: str,
        target_model: str,
        req_xlate: float,
    ) -> None:
        start = time.monotonic()
        usage: dict[str, Any] = {}
        stop_reason: str | None = None
        backend_wait = [0.0]
        # Without this OpenAI-compatible backends omit usage from the final chunk.
        request_body.setdefault("stream_options", {"include_usage": True})
        transport = proto.response_stream(200, SSE_HEADERS)
        try:
            chunks: AsyncIterator[Any] = client.stream(cast(Any, request_body))
            if timings:
                chunks = _timed_chunks(chunks, backend_wait)
            events = stream_to_anthropic(chunks, model=requested_model)
            async for name, payload in events:
                if name == "message_delta":
                    usage = payload.get("usage") or {}
                    stop_reason = (payload.get("delta") or {}).get("stop_reason")
                await transport.send_str(_sse(name, payload))
        except Exception as e:  # error mid-stream: report in-band, Anthropic style
            log.warning(
                "%s → %s | stream failed after %.2fs: %s",
                requested_model,
                target_model,
                time.monotonic() - start,
                e,
            )
            _, body = error_to_anthropic(e)
            await transport.send_str(_sse("error", body))
            return
        elapsed = time.monotonic() - start
        _log_request(
            requested_model,
            target_model,
            kind="stream",
            usage=usage,
            stop_reason=stop_reason,
            elapsed=elapsed,
            timing=_timing_detail(elapsed + req_xlate, backend_wait[0], req_xlate)
            if timings
            else "",
        )

    @router.post("/v1/messages")
    async def messages(scope: RSGIScope, proto: RSGIHTTPProtocol) -> Response | None:
        body = cast("MessagesBody", _json_loads(await proto()))
        requested_model = body["model"]
        target = _pick_model(requested_model, model=model, small_model=small_model)
        # Image-bearing requests need a multimodal backend, whatever the tier.
        if vision_model and _has_images(cast(list, body["messages"])):
            target = vision_model
        # Anthropic clients ask for large budgets (32k); backends cap lower.
        body["max_tokens"] = min(body["max_tokens"], max_output_tokens)
        xlate_start = time.perf_counter()
        openai_body = messages_to_openai(body, model=target)
        req_xlate = time.perf_counter() - xlate_start
        if body.get("stream"):
            await _stream(
                proto,
                cast("dict[str, Any]", openai_body),
                requested_model,
                target,
                req_xlate,
            )
            return None
        start = time.monotonic()
        try:
            data, _ = await client.complete(openai_body)
        except Exception as e:
            log.warning(
                "%s → %s | request failed after %.2fs: %s",
                requested_model,
                target,
                time.monotonic() - start,
                e,
            )
            status, error_body = error_to_anthropic(e)
            return Response(error_body, status=status)
        backend_s = time.monotonic() - start
        resp = response_to_anthropic(data, model=requested_model)
        elapsed = time.monotonic() - start + req_xlate
        _log_request(
            requested_model,
            target,
            kind="complete",
            usage=cast("dict[str, Any]", resp.get("usage") or {}),
            stop_reason=resp.get("stop_reason"),
            elapsed=elapsed,
            timing=_timing_detail(elapsed, backend_s, req_xlate) if timings else "",
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
    host: str = Option("127.0.0.1", "--host", help="Bind address"),
    port: int = Option(4000, "-p", "--port", help="Port to listen on"),
    trace: bool = Option(
        False, "--trace", help="Instrument proxied requests (Langfuse or OTLP export)"
    ),
    verbose: bool = Option(
        False,
        "-v",
        "--verbose",
        help="Log each proxied request (models, tokens, duration)",
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
    # Validate configuration in the CLI process for clean errors; workers
    # rebuild the client from the env snapshot below.
    _make_client(backend_url, model, api_key_env)

    env: dict[str, str | None] = {
        "BACKEND_URL": backend_url,
        "MODEL": model,
        "SMALL_MODEL": small_model,
        "VISION_MODEL": vision_model,
        "API_KEY_ENV": api_key_env,
        "MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "TRACE": "1" if trace else "",
        "VERBOSE": "1" if (verbose or timings) else "",
        "TIMINGS": "1" if timings else "",
    }
    for key, value in env.items():
        if value:
            os.environ[ENV_PREFIX + key] = value
        else:
            os.environ.pop(ENV_PREFIX + key, None)

    console.print(
        f"[green]Anthropic-compatible proxy on http://{host}:{port} "
        f"→ {backend_url or os.environ.get(PADWAN_BASE_URL_ENV)} "
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
