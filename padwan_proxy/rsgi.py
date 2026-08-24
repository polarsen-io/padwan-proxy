# Granian RSGI target: rebuilds the proxy from the env snapshot written by
# proxy_command, once per worker process.
import logging
import os

from gravier import App

from .proxy import ENV_PREFIX, _make_client, build_router


def _env(key: str) -> str | None:
    return os.environ.get(ENV_PREFIX + key) or None


if _env("VERBOSE"):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
if _env("TRACE"):
    from .trace import enable_tracing

    enable_tracing()

_client = _make_client(_env("BACKEND_URL"), _env("MODEL") or "", _env("API_KEY_ENV"))


async def _open_client() -> None:
    await _client.__aenter__()


async def _close_client() -> None:
    await _client.__aexit__(None, None, None)


app = App(
    build_router(
        client=_client,
        model=_env("MODEL") or "",
        small_model=_env("SMALL_MODEL"),
        vision_model=_env("VISION_MODEL"),
        max_output_tokens=int(_env("MAX_OUTPUT_TOKENS") or 16384),
        timings=bool(_env("TIMINGS")),
    ),
    on_startup=[_open_client],
    on_shutdown=[_close_client],
)
