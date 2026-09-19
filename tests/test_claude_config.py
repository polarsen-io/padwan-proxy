import json

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
        pytest.param("https://api.z.ai/api/anthropic", "Z.ai", id="known_host_dotted"),
        pytest.param("https://llm.internal:8443/v1", "llm.internal", id="unknown_host"),
        pytest.param(None, None, id="no_backend_url"),
    ],
)
def test_backend_vendor(backend_url, expected):
    assert backend_vendor(backend_url) == expected


def test_fresh_dir(tmp_path):
    env = _env(_write(tmp_path / "nested"))
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.2"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.6-35b-a3b"
    assert env["API_TIMEOUT_MS"] == "3600000"
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


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("{", id="truncated_json"),
        pytest.param('["a"]', id="not_an_object"),
        pytest.param('{"env": []}', id="non_object_env"),
    ],
)
def test_refuses_to_overwrite_unreadable_settings(tmp_path, content):
    settings = tmp_path / "settings.json"
    settings.write_text(content)
    with pytest.raises(CommandError):
        _write(tmp_path)
    assert settings.read_text() == content
