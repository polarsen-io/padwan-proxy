import json
import shlex
import sys

import pytest
from piou import CommandError

from padwan_proxy.claude_config import backend_vendor, write_claude_config

SCALEWAY = "https://api.scaleway.ai/v1/"


def _write(config_dir, **overrides):
    kwargs = {
        "host": "127.0.0.1",
        "port": 4000,
        "model": "glm-5.2",
        "small_model": "qwen3.6-35b-a3b",
        "backend_url": SCALEWAY,
        "timeout": 3600.0,
        "context_window": None,
    } | overrides
    return write_claude_config(config_dir, **kwargs)


def _env(path):
    return json.loads(path.read_text())["env"]


@pytest.mark.parametrize(
    ("backend_url", "expected"),
    [
        pytest.param(SCALEWAY, "Scaleway", id="known_host"),
        pytest.param("https://llm.internal:8443/v1", "llm.internal", id="unknown_host"),
        pytest.param(None, None, id="no_backend_url"),
    ],
)
def test_backend_vendor(backend_url, expected):
    assert backend_vendor(backend_url) == expected


def test_fresh_dir(tmp_path):
    env = _env(_write(tmp_path / "nested"))
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.2"
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "ai.vendor=Scaleway"
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env


@pytest.mark.parametrize(
    ("overrides", "key", "expected"),
    [
        pytest.param(
            {"context_window": 256000},
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
            "256000",
            id="context_window_emitted",
        ),
        pytest.param(
            {"small_model": None},
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "glm-5.2",
            id="no_small_model_falls_back_to_main",
        ),
        pytest.param(
            {"host": "0.0.0.0", "port": 8080},
            "ANTHROPIC_BASE_URL",
            "http://127.0.0.1:8080",
            id="wildcard_bind_is_dialable",
        ),
        pytest.param(
            {"timeout": 1.5}, "API_TIMEOUT_MS", "1500", id="timeout_in_milliseconds"
        ),
    ],
)
def test_emitted_values(tmp_path, overrides, key, expected):
    assert _env(_write(tmp_path, **overrides))[key] == expected


def test_preserves_unrelated_settings(tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["mcp__argent"], "defaultMode": "auto"},
                "hooks": {"Notification": []},
                "env": {"ENABLE_TOOL_SEARCH": "true", "ANTHROPIC_MODEL": "stale"},
            }
        )
    )
    settings = json.loads(_write(tmp_path).read_text())
    assert settings["permissions"]["allow"] == ["mcp__argent"]
    assert settings["hooks"] == {"Notification": []}
    assert settings["env"]["ENABLE_TOOL_SEARCH"] == "true"
    assert settings["env"]["ANTHROPIC_MODEL"] == "glm-5.2"


def test_merges_otel_attributes(tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "OTEL_RESOURCE_ATTRIBUTES": (
                        "device.name=julien-lenovo,ai.route=pglm,ai.vendor=stale"
                    )
                }
            }
        )
    )
    assert (
        _env(_write(tmp_path))["OTEL_RESOURCE_ATTRIBUTES"]
        == "device.name=julien-lenovo,ai.route=pglm,ai.vendor=Scaleway"
    )


def test_approvals_enable_is_idempotent_and_preserves_unrelated_settings(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text("TYPESAFE_API_KEY=private-test-key\n")
    monkeypatch.setenv("PADWAN_PROXY_APPROVALS_ENV_FILE", str(env_file))
    custom_hook = {"type": "command", "command": "check-local-policy"}
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"], "defaultMode": "auto"},
                "hooks": {
                    "PreToolUse": [{"matcher": "*", "hooks": [custom_hook]}],
                    "Notification": [{"hooks": [custom_hook]}],
                },
                "env": {"KEEP": "yes"},
            }
        )
    )

    path = _write(tmp_path, approvals=True)
    first = path.read_text()
    settings = json.loads(first)
    _write(tmp_path, approvals=True)

    assert path.read_text() == first
    assert settings["env"]["PADWAN_PROXY_APPROVALS_ENV_FILE"] == str(env_file)
    assert "private-test-key" not in first
    assert settings["permissions"] == {"allow": ["Read"], "defaultMode": "auto"}
    assert settings["env"]["KEEP"] == "yes"
    assert settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    assert settings["hooks"]["Notification"] == [{"hooks": [custom_hook]}]
    assert settings["hooks"]["PreToolUse"] == [
        {
            "matcher": "*",
            "hooks": [
                custom_hook,
                {
                    "type": "command",
                    "command": shlex.join(
                        [sys.executable, "-I", "-m", "padwan_proxy.approvals"]
                    ),
                    "timeout": 20,
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("check-local-policy", id="custom_hook"),
        pytest.param(
            "echo audit -m padwan_proxy.approvals", id="similar_custom_command"
        ),
    ],
)
def test_approvals_disable_removes_only_managed_hook(tmp_path, command):
    custom_hook = {"type": "command", "command": command}
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"], "defaultMode": "auto"},
                "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [custom_hook]}]},
                "env": {"KEEP": "yes"},
            }
        )
    )
    _write(tmp_path, approvals=True)

    settings = json.loads(_write(tmp_path, approvals=False).read_text())

    assert settings["hooks"]["PreToolUse"] == [{"matcher": "*", "hooks": [custom_hook]}]
    assert settings["permissions"] == {"allow": ["Read"], "defaultMode": "auto"}
    assert settings["env"]["KEEP"] == "yes"
    assert settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"


@pytest.mark.parametrize(
    ("content", "approvals"),
    [
        pytest.param('{"hooks": []}', True, id="hooks_not_object"),
        pytest.param(
            '{"hooks": {"PreToolUse": {}}}', True, id="pre_tool_use_not_array"
        ),
        pytest.param('{"permissions": []}', True, id="permissions_not_object"),
        pytest.param("{", None, id="truncated_json"),
        pytest.param('["a"]', None, id="not_an_object"),
        pytest.param('{"env": []}', None, id="non_object_env"),
    ],
)
def test_refuses_to_overwrite_unreadable_or_malformed_settings(
    tmp_path, content, approvals
):
    settings = tmp_path / "settings.json"
    settings.write_text(content)
    with pytest.raises(CommandError):
        _write(tmp_path, approvals=approvals)
    assert settings.read_text() == content


_URL = ("approvals_url", "http://127.0.0.1:9999", "PADWAN_PROXY_APPROVALS_URL")
_GATE = ("approvals_confidence", 0.5, "PADWAN_PROXY_APPROVALS_MIN_CONFIDENCE")


@pytest.mark.parametrize(
    "seed, overrides, expected",
    [
        pytest.param(
            _URL,
            {"approvals": True, "approvals_url": "http://127.0.0.1:4000"},
            "http://127.0.0.1:4000",
            id="url_local_model",
        ),
        # A stale URL would aim the hook at a proxy no longer serving Laya.
        pytest.param(_URL, {"approvals": True}, None, id="url_hosted_api_clears_it"),
        pytest.param(_URL, {"approvals": False}, None, id="url_disabled_clears_it"),
        pytest.param(_URL, {}, "http://127.0.0.1:9999", id="url_unmanaged"),
        pytest.param(
            _GATE,
            {"approvals": True, "approvals_confidence": 0.15},
            "0.15",
            id="gate_set",
        ),
        # A stale gate would keep the local threshold on the hosted model.
        pytest.param(_GATE, {"approvals": True}, None, id="gate_cleared"),
        pytest.param(_GATE, {}, "0.5", id="gate_untouched_when_unmanaged"),
    ],
)
def test_local_model_env_tracks_approvals(tmp_path, seed, overrides, expected):
    arg, value, key = seed
    _write(tmp_path, approvals=True, **{arg: value})
    assert _env(_write(tmp_path, **overrides)).get(key) == expected


@pytest.mark.parametrize(
    "approvals",
    [
        pytest.param(None, id="preserve"),
        pytest.param(True, id="enable"),
        pytest.param(False, id="disable"),
    ],
)
def test_legacy_approval_hook_migration(tmp_path, approvals):
    custom = {"type": "command", "command": "echo audit -m padwan_proxy.jev"}
    legacy = {
        "type": "command",
        "command": shlex.join([sys.executable, "-I", "-m", "padwan_proxy.jev"]),
        "timeout": 15,
    }
    permissions = {"allow": ["Read"], "defaultMode": "auto"}
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "permissions": permissions,
                "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [custom, legacy]}]},
                "env": {
                    "KEEP": "yes",
                    "PADWAN_PROXY_JEV_LOG": "/old/jev.jsonl",
                    "PADWAN_PROXY_JEV_URL": "http://127.0.0.1:4000",
                    "PADWAN_PROXY_JEV_ENV_FILE": "/private/key.env",
                    "PADWAN_PROXY_JEV_MIN_CONFIDENCE": "0.99",
                },
            }
        )
    )
    path = _write(tmp_path, approvals=approvals)
    first = path.read_text()
    settings = json.loads(first)
    hooks = settings["hooks"]["PreToolUse"][0]["hooks"]
    assert hooks[0] == custom
    assert settings["permissions"] == permissions
    assert len(hooks) == (1 if approvals is False else 2)
    if approvals is not False:
        assert shlex.split(hooks[1]["command"])[-1] == "padwan_proxy.approvals"
    if approvals is None:
        assert hooks[1]["timeout"] == 15
        assert settings["env"]["PADWAN_PROXY_APPROVALS_ENV_FILE"] == "/private/key.env"
        assert settings["env"]["PADWAN_PROXY_APPROVALS_MIN_CONFIDENCE"] == "0.99"
    assert settings["env"]["KEEP"] == "yes"
    assert not any(key.startswith("PADWAN_PROXY_JEV_") for key in settings["env"])
    _write(tmp_path, approvals=approvals)
    assert path.read_text() == first
