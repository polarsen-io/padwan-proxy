# Granian RSGI target: rebuilds the proxy from the env snapshot written by
# proxy_command, once per worker process.
from gravier import App

from . import env
from .logs import setup_logging
from .proxy import _make_client, build_router
from .systemone import load as load_laya

_client = _make_client(env.BACKEND_URL, env.MODEL, env.API_KEY_ENV, timeout=env.TIMEOUT)
# Load and warm the model before accepting hook calls.
_laya = (
    load_laya(env.LAYA_MODEL, subfolder=env.LAYA_SUBFOLDER) if env.LAYA_MODEL else None
)


async def _startup() -> None:
    if env.VERBOSE:
        setup_logging(rich=env.RICH)
    if env.TRACE:
        from .trace import enable_tracing

        enable_tracing(capture_content=env.TRACE_CONTENT)
    await _client.__aenter__()


async def _shutdown() -> None:
    await _client.__aexit__(None, None, None)


app = App(
    build_router(
        client=_client,
        model=env.MODEL,
        small_model=env.SMALL_MODEL,
        vision_model=env.VISION_MODEL,
        vision_mode=env.VISION_MODE,
        max_output_tokens=env.MAX_OUTPUT_TOKENS,
        stream_retries=env.STREAM_RETRIES,
        timings=env.TIMINGS,
        breakdown=env.BREAKDOWN,
        rich=env.RICH,
        laya=_laya,
    ),
    on_startup=[_startup],
    on_shutdown=[_shutdown],
)
