import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from padwan_proxy.claude_config import write_claude_config

from .conftest import Provider

pytestmark = pytest.mark.e2e

_CONFIGURED_ENV = (
    "PADWAN_PROXY_E2E_BASE_URL",
    "PADWAN_PROXY_E2E_MODEL",
)


def _configured_provider() -> Provider:
    api_key_env = (
        "PADWAN_PROXY_E2E_API_KEY"
        if os.environ.get("PADWAN_PROXY_E2E_API_KEY")
        else "PADWAN_API_KEY"
    )
    return Provider(
        os.environ.get("PADWAN_PROXY_E2E_BASE_URL", ""),
        os.environ.get("PADWAN_PROXY_E2E_MODEL", ""),
        api_key_env,
    )


def _blocks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for record in records:
        message = record.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            blocks.extend(
                block for block in message["content"] if isinstance(block, dict)
            )
        event = record.get("event")
        if isinstance(event, dict):
            content_block = event.get("content_block")
            delta = event.get("delta")
            if isinstance(content_block, dict):
                blocks.append(content_block)
            if isinstance(delta, dict):
                blocks.append(delta)
    return blocks


@pytest.mark.parametrize(
    "live_proxy",
    [
        pytest.param(
            _configured_provider(),
            id="configured",
            marks=[
                pytest.mark.skipif(
                    any(not os.environ.get(name) for name in _CONFIGURED_ENV)
                    or not (
                        os.environ.get("PADWAN_PROXY_E2E_API_KEY")
                        or os.environ.get("PADWAN_API_KEY")
                    ),
                    reason="configured provider environment is incomplete",
                ),
                pytest.mark.skipif(
                    shutil.which("claude") is None, reason="claude CLI not installed"
                ),
            ],
        )
    ],
    indirect=True,
)
def test_claude_code_reads_isolated_file(
    live_proxy: tuple[str, Provider], tmp_path: Path
) -> None:
    base_url, provider = live_proxy
    parsed_url = urlparse(base_url)
    if parsed_url.port is None:
        raise AssertionError(f"proxy URL has no port: {base_url}")

    config_dir = tmp_path / "config"
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    token = f"probe-{secrets.token_hex(4)}"
    (work_dir / "fact.txt").write_text(token)
    write_claude_config(
        config_dir,
        host=parsed_url.hostname or "127.0.0.1",
        port=parsed_url.port,
        model=provider.model,
        small_model=None,
        backend_url=provider.base_url,
        timeout=300,
    )

    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env.pop("CLAUDECODE", None)
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--restricted",
            "--settings",
            str(config_dir / "settings.json"),
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--no-chrome",
            "--no-session-persistence",
            "Read fact.txt with the Read tool, then reply with its exact contents.",
        ],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    stdout_path = tmp_path / "claude.stdout.jsonl"
    stderr_path = tmp_path / "claude.stderr.log"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    assert result.returncode == 0, (
        f"claude failed; stdout={stdout_path}, stderr={stderr_path}"
    )
    records = [json.loads(line) for line in result.stdout.splitlines() if line]
    blocks = _blocks(records)
    assert any(block.get("type") in {"thinking", "thinking_delta"} for block in blocks)
    assert any(
        block.get("type") == "tool_use" and block.get("name") == "Read"
        for block in blocks
    )
    assert any(block.get("type") == "tool_result" for block in blocks)
    final = next(record for record in records if record.get("type") == "result")
    assert final.get("is_error") is False
    assert token in str(final.get("result", ""))
