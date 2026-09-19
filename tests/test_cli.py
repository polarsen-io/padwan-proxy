import json
import os
from unittest.mock import Mock, patch

import pytest
from padwan_llm.client import PADWAN_BASE_URL_ENV
from piou import Cli

from padwan_proxy import proxy


@pytest.mark.parametrize(
    "config_exists",
    [
        pytest.param(True, id="existing_config_preserved"),
        pytest.param(False, id="new_config_created"),
    ],
)
def test_claude_config_cli_uses_resolved_options_and_preserves_settings(
    tmp_path, monkeypatch, config_exists
):
    config_dir = tmp_path / "new-config"
    if config_exists:
        config_dir.mkdir()
        (config_dir / "settings.json").write_text(
            json.dumps({"permissions": {"allow": ["Read"]}, "env": {"KEEP": "yes"}})
        )
    monkeypatch.setenv(PADWAN_BASE_URL_ENV, "https://api.z.ai/api/coding/paas/v4")
    monkeypatch.setattr(proxy, "_make_client", Mock())
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with patch.dict(os.environ):
        cli.run_with_args(
            "-m",
            "glm-5",
            "--small-model",
            "glm-4.5-air",
            "--host",
            "0.0.0.0",
            "--port",
            "4100",
            "--timeout",
            "90.5",
            "--claude-config",
            str(config_dir),
            "--context-window",
            "202752",
        )

    settings = json.loads((config_dir / "settings.json").read_text())
    if config_exists:
        assert settings["permissions"] == {"allow": ["Read"]}
        assert settings["env"]["KEEP"] == "yes"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4100"
    assert settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5"
    assert settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-4.5-air"
    assert settings["env"]["API_TIMEOUT_MS"] == "90500"
    assert settings["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "202752"
    assert settings["env"]["OTEL_RESOURCE_ATTRIBUTES"] == "ai.vendor=Z.ai"
    serve.assert_called_once_with("padwan_proxy.rsgi:app", host="0.0.0.0", port=4100)


def test_claude_config_cli_rejects_nonpositive_context_window(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "_make_client", Mock())
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with patch.dict(os.environ), pytest.raises(SystemExit):
        cli.run_with_args(
            "--backend-url",
            "https://api.example.com/v1",
            "-m",
            "model",
            "--claude-config",
            str(tmp_path / "config"),
            "--context-window",
            "0",
        )

    serve.assert_not_called()
    assert not (tmp_path / "config").exists()


@pytest.mark.parametrize(
    ("rich", "expects_markup"),
    [
        pytest.param(False, False, id="plain"),
        pytest.param(True, True, id="rich"),
    ],
)
def test_rich_logging_toggles_markup(rich, expects_markup):
    from padwan_proxy import logs

    try:
        logs.setup_logging(rich=rich)
        assert ("[bold cyan]" in logs._style("glm-5", "bold cyan")) is expects_markup
    finally:
        logs.setup_logging()
