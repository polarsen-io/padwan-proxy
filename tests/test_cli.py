import json
import os
from unittest.mock import Mock, patch

import pytest
from piou import Cli

from padwan_proxy import proxy
from padwan_proxy.defaults import PADWAN_BASE_URL_ENV


@pytest.fixture(autouse=True)
def free_port(monkeypatch):
    """The real probe would hit whatever is serving :4000 on the test machine."""
    monkeypatch.setattr(proxy, "_port_answers", lambda host, port: False)


def _cli(monkeypatch, *, installed=True):
    make_client, serve = Mock(), Mock()
    monkeypatch.setattr(proxy, "_make_client", make_client)
    monkeypatch.setattr(
        proxy, "find_spec", lambda name: object() if installed else None
    )
    monkeypatch.setattr(proxy, "serve", serve)
    cli = Cli()
    cli.main()(proxy.proxy_command)
    return cli, serve, make_client


_BACKEND = ("--backend-url", "https://api.example.com/v1", "-m", "model")


@pytest.mark.parametrize(
    "flags, hook_kept",
    [
        pytest.param(["--no-approvals"], False, id="disabled_removes_hook"),
        pytest.param([], True, id="unset_leaves_hook"),
    ],
)
def test_approvals_flag_maps_to_the_hook(tmp_path, monkeypatch, flags, hook_kept):
    cli, _, _ = _cli(monkeypatch)
    with patch.dict(os.environ):
        cli.run_with_args(
            *_BACKEND, "--claude-config", str(tmp_path), "--approvals", "jev"
        )
        cli.run_with_args(*_BACKEND, "--claude-config", str(tmp_path), *flags)
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert bool(settings["hooks"]["PreToolUse"][0]["hooks"]) is hook_kept


def test_claude_config_cli_uses_resolved_options(tmp_path, monkeypatch):
    config_dir = tmp_path / "new-config"
    monkeypatch.setenv(PADWAN_BASE_URL_ENV, "https://api.z.ai/api/coding/paas/v4")
    cli, serve, _ = _cli(monkeypatch)

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
    assert "defaultMode" not in settings.get("permissions", {})
    assert settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    assert settings["env"]["PADWAN_PROXY_APPROVALS_LOG"] == str(
        config_dir / "approvals.jsonl"
    )
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "*"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4100"
    assert settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5"
    assert settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-4.5-air"
    assert settings["env"]["API_TIMEOUT_MS"] == "90500"
    assert settings["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "202752"
    assert settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5"
    assert settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-4.5-air"
    assert settings["env"]["API_TIMEOUT_MS"] == "90500"
    assert settings["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "202752"
    assert settings["env"]["OTEL_RESOURCE_ATTRIBUTES"] == "ai.vendor=Z.ai"
    serve.assert_called_once_with("padwan_proxy.rsgi:app", host="0.0.0.0", port=4100)


@pytest.mark.parametrize(
    "flags, installed, port_busy",
    [
        pytest.param(
            ["--context-window", "0"], True, False, id="nonpositive_context_window"
        ),
        pytest.param(
            ["--approvals", "jev", "--no-claude-config"],
            True,
            False,
            id="approvals_need_claude_config",
        ),
        pytest.param(["--approvals", "laya"], False, False, id="laya_extra_missing"),
        pytest.param(
            ["--approval-confidence", "0.15"],
            True,
            False,
            id="confidence_needs_approvals",
        ),
        pytest.param(
            ["--approval-model", "jev-preview"], True, False, id="model_needs_approvals"
        ),
        pytest.param(
            ["--approvals", "jev", "--approval-subfolder", "multilingual"],
            True,
            False,
            id="subfolder_needs_laya",
        ),
        pytest.param(
            ["--approvals", "jev", "--approval-confidence", "0"],
            True,
            False,
            id="confidence_zero",
        ),
        # Granian shares the port instead of failing, so two configs would flap.
        pytest.param([], True, True, id="port_already_served"),
    ],
)
def test_cli_rejects_invalid_flags(tmp_path, monkeypatch, flags, installed, port_busy):
    monkeypatch.setattr(proxy, "_port_answers", lambda host, port: port_busy)
    cli, serve, make_client = _cli(monkeypatch, installed=installed)
    config = tmp_path / "config"
    no_config = "--no-claude-config" in flags
    argv = [f for f in flags if f != "--no-claude-config"]
    if not no_config:
        argv += ["--claude-config", str(config)]

    with patch.dict(os.environ), pytest.raises(SystemExit):
        cli.run_with_args(*_BACKEND, *argv)

    serve.assert_not_called()
    assert not (config / "settings.json").exists()
    if no_config:
        make_client.assert_not_called()


@pytest.mark.parametrize(
    "flags, installed, laya, hook_env",
    [
        pytest.param(
            [
                "--approvals",
                "laya",
                "--approval-model",
                "convaiinnovations/laya",
                "--approval-subfolder",
                "multilingual",
            ],
            True,
            {
                "PADWAN_PROXY_LAYA_MODEL": "convaiinnovations/laya",
                "PADWAN_PROXY_LAYA_SUBFOLDER": "multilingual",
            },
            {
                "PADWAN_PROXY_APPROVALS_URL": "http://127.0.0.1:4000",
                "PADWAN_PROXY_APPROVALS_MODEL": None,
            },
            id="laya",
        ),
        pytest.param(
            ["--approvals", "jev", "--approval-model", "jev-preview"],
            True,
            {"PADWAN_PROXY_LAYA_MODEL": None},
            {
                "PADWAN_PROXY_APPROVALS_MODEL": "jev-preview",
                "PADWAN_PROXY_APPROVALS_URL": None,
            },
            id="jev",
        ),
    ],
)
def test_hook_routing_per_backend(
    tmp_path, monkeypatch, flags, installed, laya, hook_env
):
    cli, serve, _ = _cli(monkeypatch, installed=installed)
    with patch.dict(os.environ):
        cli.run_with_args(*_BACKEND, "--claude-config", str(tmp_path), *flags)
        for key, value in laya.items():
            assert os.environ.get(key) == value
    env = json.loads((tmp_path / "settings.json").read_text())["env"]
    for key, value in hook_env.items():
        assert env.get(key) == value
    serve.assert_called_once()


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
