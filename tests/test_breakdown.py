from typing import Any, cast

import pytest

from padwan_proxy.breakdown import PromptBreakdown, format_breakdown, prompt_breakdown


def _tool(name: str, filler: str = "") -> dict:
    return {
        "name": name,
        "description": f"does something{filler}",
        "input_schema": {"type": "object", "properties": {}},
    }


@pytest.mark.parametrize(
    "body, expected_count, expected_sources",
    [
        pytest.param({"messages": []}, 0, (), id="no_tools_no_system"),
        pytest.param(
            {"messages": [], "tools": [_tool("Read"), _tool("Bash")]},
            2,
            ("builtin",),
            id="builtin_tools_grouped_together",
        ),
        pytest.param(
            {
                "messages": [],
                "tools": [
                    _tool("mcp__notion-obitrain__search", filler="x" * 400),
                    _tool("mcp__argent__describe"),
                    _tool("Read"),
                ],
            },
            3,
            ("notion-obitrain", "argent", "builtin"),
            id="mcp_servers_ranked_by_size",
        ),
        pytest.param(
            {"messages": [], "tools": [_tool("mcp__claude_ai_Gmail__send_message")]},
            1,
            ("claude_ai_Gmail",),
            id="server_name_containing_underscores",
        ),
    ],
)
def test_prompt_breakdown(body, expected_count, expected_sources):
    result = prompt_breakdown(body)
    assert result.tool_count == expected_count
    assert tuple(name for name, _ in result.by_source) == expected_sources
    assert result.tools == sum(size for _, size in result.by_source)


@pytest.mark.parametrize(
    "system, expected_system",
    [
        pytest.param(None, 0, id="absent"),
        pytest.param("you are helpful", 4, id="plain_string"),
        pytest.param(
            [{"type": "text", "text": "you are helpful"}], 10, id="content_blocks"
        ),
    ],
)
def test_system_sizing(system, expected_system):
    body: dict[str, Any] = {"messages": []}
    if system is not None:
        body["system"] = system
    assert prompt_breakdown(cast("Any", body)).system == expected_system


@pytest.mark.parametrize(
    "breakdown, expected",
    [
        pytest.param(
            PromptBreakdown(
                system=3900,
                tools=186_000,
                tool_count=312,
                messages=47_000,
                by_source=(("notion", 41_000), ("argent", 38_000)),
            ),
            "sys=3.9k tools=186k(312) msgs=47k | top: notion 41k, argent 38k",
            id="thousands_abbreviated",
        ),
        pytest.param(
            PromptBreakdown(
                system=0, tools=0, tool_count=0, messages=812, by_source=()
            ),
            "sys=0 tools=0(0) msgs=812",
            id="no_tools_omits_top",
        ),
    ],
)
def test_format_breakdown(breakdown, expected):
    assert format_breakdown(breakdown) == expected


def test_format_breakdown_caps_the_source_list():
    breakdown = PromptBreakdown(
        system=0,
        tools=4,
        tool_count=4,
        messages=0,
        by_source=(("a", 4), ("b", 3), ("c", 2), ("d", 1)),
    )
    assert format_breakdown(breakdown, top=2).endswith("| top: a 4, b 3")
