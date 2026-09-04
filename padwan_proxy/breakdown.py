from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from padwan_llm._json import dumps as _json_dumps

if TYPE_CHECKING:
    from padwan_llm.anthropic.models import MessagesBody

__all__ = ("PromptBreakdown", "format_breakdown", "prompt_breakdown")

_MCP_PREFIX = "mcp__"
_BUILTIN = "builtin"


def _tokens(obj: object) -> int:
    """Chars/4 estimate, as count_tokens does: no backend tokenizer available here."""
    return len(_json_dumps(obj)) // 4


def _source(tool_name: str) -> str:
    """MCP server a tool comes from, or `builtin` for the client's own tools."""
    if not tool_name.startswith(_MCP_PREFIX):
        return _BUILTIN
    return tool_name.removeprefix(_MCP_PREFIX).split("__", 1)[0]


@dataclass(frozen=True, slots=True)
class PromptBreakdown:
    """Estimated token split of an incoming Messages body, `by_source` descending."""

    system: int
    tools: int
    tool_count: int
    messages: int
    by_source: tuple[tuple[str, int], ...]


def prompt_breakdown(body: MessagesBody) -> PromptBreakdown:
    """Size each section of a Messages body, grouping tool schemas by MCP server."""
    by_source: dict[str, int] = {}
    tools = body.get("tools") or []
    for tool in tools:
        source = _source(tool.get("name", ""))
        by_source[source] = by_source.get(source, 0) + _tokens(tool)
    system = body.get("system")
    return PromptBreakdown(
        system=_tokens(system) if system else 0,
        tools=sum(by_source.values()),
        tool_count=len(tools),
        messages=_tokens(body.get("messages") or []),
        by_source=tuple(sorted(by_source.items(), key=lambda kv: -kv[1])),
    )


def _fmt(tokens: int) -> str:
    if tokens >= 10_000:
        return f"{tokens / 1000:.0f}k"
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)


def format_breakdown(breakdown: PromptBreakdown, *, top: int = 3) -> str:
    """One-line summary: section sizes, then the `top` heaviest tool sources."""
    line = (
        f"sys={_fmt(breakdown.system)} "
        f"tools={_fmt(breakdown.tools)}({breakdown.tool_count}) "
        f"msgs={_fmt(breakdown.messages)}"
    )
    if sources := breakdown.by_source[:top]:
        listed = ", ".join(f"{name} {_fmt(size)}" for name, size in sources)
        line += f" | top: {listed}"
    return line
