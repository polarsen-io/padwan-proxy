import logging
from typing import Any

log = logging.getLogger("padwan_proxy")


def setup_logging() -> None:
    """Configure per-request logging for `-v`/`-vv`."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def log_request(
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


def timing_detail(elapsed: float, backend: float, req_xlate: float) -> str:
    """Format the -vv timing segment: backend wait vs time spent in the proxy."""
    overhead = max(0.0, elapsed - backend - req_xlate)
    return (
        f" (backend {backend:.2f}s, req-xlate {req_xlate * 1e3:.1f}ms, "
        f"proxy {overhead * 1e3:.1f}ms)"
    )
