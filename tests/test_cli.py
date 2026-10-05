import json
import os
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
from padwan_ai.client import PADWAN_BASE_URL_ENV
from piou import Cli

from padwan_proxy import proxy


@pytest.fixture(autouse=True)
def free_port(monkeypatch):
    """The real probe would hit whatever is serving :4000 on the test machine."""
    monkeypatch.setattr(proxy, "_port_answers", lambda host, port: False)


@pytest.mark.parametrize(
    "flags, status",
    [
        pytest.param(["--approvals", "jev"], "jev over 0.99", id="enabled"),
        pytest.param(["--approvals", "laya"], "laya local advisory", id="laya_gate"),
        pytest.param(
            ["--approvals", "laya", "--approval-confidence", "0.15"],
            "advisory (always ask)",
            id="lowered_gate",
        ),
        pytest.param(["--no-approvals"], "hook disabled", id="disabled"),
        pytest.param([], "hook unchanged", id="preserved"),
    ],
)
def test_startup_shows_approval_model(tmp_path, monkeypatch, capsys, flags, status):
    monkeypatch.setattr(proxy, "_make_client", Mock())
    monkeypatch.setattr(proxy, "find_spec", lambda name: object())
    monkeypatch.setattr(proxy, "serve", Mock())
    cli = Cli()
    cli.main()(proxy.proxy_command)
    with patch.dict(os.environ):
        cli.run_with_args("-m", "glm-5.2", "--claude-config", str(tmp_path), *flags)
    output = capsys.readouterr().out
    assert "approvals" in output
    assert status in output


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
            "--approvals",
            "jev",
        )

    settings = json.loads((config_dir / "settings.json").read_text())
    if config_exists:
        assert settings["permissions"]["allow"] == ["Read"]
        assert settings["env"]["KEEP"] == "yes"
    assert "defaultMode" not in settings.get("permissions", {})
    assert settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    assert settings["env"]["PADWAN_PROXY_APPROVALS_LOG"] == str(
        config_dir / "approvals.jsonl"
    )
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "*"
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 20
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
    "flags",
    [
        pytest.param(["--approvals", "jev"], id="enable"),
        pytest.param(["--no-approvals"], id="disable"),
    ],
)
def test_approvals_cli_requires_claude_config(flags, monkeypatch):
    make_client = Mock()
    monkeypatch.setattr(proxy, "_make_client", make_client)
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with pytest.raises(SystemExit):
        cli.run_with_args(
            "--backend-url", "https://api.example.com/v1", "-m", "model", *flags
        )

    make_client.assert_not_called()
    serve.assert_not_called()


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


@pytest.mark.parametrize(
    "installed, expected_serve",
    [
        pytest.param(True, True, id="extra_installed"),
        # The worker imports laya; without it every granian worker would crash.
        pytest.param(False, False, id="extra_missing"),
    ],
)
def test_laya_wires_the_worker_and_the_hook(
    tmp_path, monkeypatch, installed, expected_serve
):
    monkeypatch.setattr(proxy, "_make_client", Mock())
    monkeypatch.setattr(
        proxy, "find_spec", lambda name: object() if installed else None
    )
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with (
        patch.dict(os.environ),
        nullcontext() if installed else pytest.raises(SystemExit),
    ):
        cli.run_with_args(
            "--backend-url",
            "https://api.example.com/v1",
            "-m",
            "glm-5.2",
            "--claude-config",
            str(tmp_path),
            "--approvals",
            "laya",
            "--approval-model",
            "convaiinnovations/laya",
            "--approval-subfolder",
            "multilingual",
        )
        model = "convaiinnovations/laya"
        assert os.environ["PADWAN_PROXY_LAYA_MODEL"] == model
        assert os.environ["PADWAN_PROXY_LAYA_SUBFOLDER"] == "multilingual"
        env = json.loads((tmp_path / "settings.json").read_text())["env"]
        assert env["PADWAN_PROXY_APPROVALS_URL"] == "http://127.0.0.1:4000"
        # The proxy holds the checkpoint; only a hosted model reaches the hook.
        assert "PADWAN_PROXY_APPROVALS_MODEL" not in env

    assert serve.called is expected_serve


def test_jev_sends_its_model_to_the_hook_and_loads_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "_make_client", Mock())
    monkeypatch.setattr(proxy, "serve", Mock())
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with patch.dict(os.environ):
        cli.run_with_args(
            "--backend-url",
            "https://api.example.com/v1",
            "-m",
            "glm-5.2",
            "--claude-config",
            str(tmp_path),
            "--approvals",
            "jev",
            "--approval-model",
            "jev-preview",
        )
        assert os.environ.get("PADWAN_PROXY_LAYA_MODEL") is None
        env = json.loads((tmp_path / "settings.json").read_text())["env"]
        assert env["PADWAN_PROXY_APPROVALS_MODEL"] == "jev-preview"
        assert "PADWAN_PROXY_APPROVALS_URL" not in env


@pytest.mark.parametrize(
    "flags, written",
    [
        pytest.param(
            ["--approvals", "jev", "--approval-confidence", "0.15"], "0.15", id="valid"
        ),
        pytest.param(["--approval-confidence", "0.15"], None, id="needs_approvals"),
        pytest.param(["--approval-model", "jev-preview"], None, id="model_needs_it"),
        pytest.param(
            ["--approvals", "jev", "--approval-subfolder", "multilingual"],
            None,
            id="subfolder_needs_laya",
        ),
        pytest.param(
            ["--approvals", "jev", "--approval-confidence", "0"], None, id="zero"
        ),
        pytest.param(
            ["--approvals", "jev", "--approval-confidence", "1.5"], None, id="over_one"
        ),
    ],
)
def test_approval_confidence_reaches_the_hook(tmp_path, monkeypatch, flags, written):
    monkeypatch.setattr(proxy, "_make_client", Mock())
    monkeypatch.setattr(proxy, "serve", Mock())
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with (
        patch.dict(os.environ),
        nullcontext() if written else pytest.raises(SystemExit),
    ):
        cli.run_with_args(
            "--backend-url",
            "https://api.example.com/v1",
            "-m",
            "glm-5.2",
            "--claude-config",
            str(tmp_path),
            *flags,
        )

    settings = tmp_path / "settings.json"
    gate = json.loads(settings.read_text())["env"] if settings.exists() else {}
    assert gate.get("PADWAN_PROXY_APPROVALS_MIN_CONFIDENCE") == written


def test_laya_serves_without_a_claude_config(monkeypatch):
    """The Docker image serves /systemone; the hook lives on another machine."""
    monkeypatch.setattr(proxy, "_make_client", Mock())
    monkeypatch.setattr(proxy, "find_spec", lambda name: object())
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with patch.dict(os.environ):
        cli.run_with_args(
            "--backend-url",
            "https://api.example.com/v1",
            "-m",
            "glm-5.2",
            "--approvals",
            "laya",
        )
        assert os.environ["PADWAN_PROXY_LAYA_MODEL"] == "convaiinnovations/laya"
    serve.assert_called_once()


@pytest.mark.parametrize(
    "answers, serves",
    [
        pytest.param(False, True, id="free"),
        # Granian shares the port instead of failing, so two configs would flap.
        pytest.param(True, False, id="already_served"),
    ],
)
def test_refuses_a_port_another_proxy_already_serves(monkeypatch, answers, serves):
    monkeypatch.setattr(proxy, "_port_answers", lambda host, port: answers)
    monkeypatch.setattr(proxy, "_make_client", Mock())
    serve = Mock()
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)

    with nullcontext() if serves else pytest.raises(SystemExit):
        cli.run_with_args("--backend-url", "https://api.example.com/v1", "-m", "model")

    assert serve.called is serves
