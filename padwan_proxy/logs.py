import logging
from collections.abc import Sequence
from typing import Any

from .utils import console

log = logging.getLogger("padwan_proxy")

# aligns the breakdown line under the message, past the "%H:%M:%S " stamp
_INDENT = " " * 9

_STOP_STYLE = {"end_turn": "green", "tool_use": "blue", "max_tokens": "yellow"}
_KIND_EMOJI = {"stream": "🌊", "complete": "📦"}

_rich = False


def setup_logging(*, rich: bool = False) -> None:
    """Configure per-request logging for `-v`/`-vv`, optionally rich-rendered."""
    global _rich
    _rich = rich
    if not rich:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        return
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                console=console,
                markup=True,
                show_path=False,
                show_level=True,
                omit_repeated_times=False,
            )
        ],
    )


def _style(text: object, style: str) -> str:
    """Rich markup when rich logging is on, plain text otherwise."""
    return f"[{style}]{text}[/{style}]" if _rich else str(text)


def route(requested: str, target: str) -> str:
    """`requested → target`, collapsed when the client asks for the backend name."""
    return target if requested == target else f"{requested} → {target}"


def _tools(names: Sequence[str], *, top: int = 4) -> str:
    """`(Read, Bash, +3)` — which tools the model asked for, truncated."""
    if not names:
        return ""
    extra = len(names) - top
    listed = ", ".join(names[:top]) + (f", +{extra}" if extra > 0 else "")
    return f"({listed})"


def _rate(usage: dict[str, Any], elapsed: float) -> str:
    """Output throughput, blank when the request was too short to be meaningful."""
    out = usage.get("output_tokens") or 0
    return f" {out / elapsed:.0f} tok/s" if out and elapsed > 0 else ""


def log_request(
    requested: str,
    target: str,
    *,
    kind: str,
    usage: dict[str, Any],
    stop_reason: str | None,
    elapsed: float,
    tools: Sequence[str] = (),
    timing: str | None = None,
    breakdown: str | None = None,
    session: str | None = None,
) -> None:
    cached = usage.get("cache_read_input_tokens")
    head = _style(route(requested, target), "bold cyan")
    if session:
        head = f"{_style(session[:8], 'dim')} {head}"
    log.info(
        "%s | %s | %s | in=%s out=%s%s | %s%s%s",
        head,
        _KIND_EMOJI.get(kind, kind) if _rich else kind,
        _style(stop_reason or "?", _STOP_STYLE.get(stop_reason or "", "yellow"))
        + _style(_tools(tools), "dim"),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        f" cached={cached}" if cached else "",
        _style(f"{elapsed:.2f}s{_rate(usage, elapsed)}", "magenta"),
        _style(timing, "dim") if timing else "",
        f"\n{'' if _rich else _INDENT}{_style(breakdown, 'dim')}" if breakdown else "",
    )


def timing_detail(elapsed: float, backend: float, req_xlate: float) -> str:
    """Format the -vv timing segment: backend wait vs time spent in the proxy."""
    overhead = max(0.0, elapsed - backend - req_xlate)
    return (
        f" (backend {backend:.2f}s, req-xlate {req_xlate * 1e3:.1f}ms, "
        f"proxy {overhead * 1e3:.1f}ms)"
    )
