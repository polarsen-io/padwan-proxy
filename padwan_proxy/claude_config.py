from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jinja2 import Environment, PackageLoader
from piou import CommandError

# Display names, because they land on dashboards as the provider label.
VENDORS = {
    "api.anthropic.com": "Anthropic",
    "api.mistral.ai": "Mistral",
    "api.moonshot.ai": "Moonshot",
    "api.openai.com": "OpenAI",
    "api.scaleway.ai": "Scaleway",
    "api.z.ai": "Z.ai",
}

SETTINGS_FILE = "settings.json"
APPROVALS_MODULE = "padwan_proxy.approvals"

_env = Environment(loader=PackageLoader("padwan_proxy", "templates"), autoescape=False)


def backend_vendor(backend_url: str | None) -> str | None:
    """Display name for the backend host: `api.z.ai` -> `Z.ai`, unknown hosts as-is."""
    host = urlparse(backend_url).hostname if backend_url else None
    return VENDORS.get(host, host) if host else None


def client_base_url(host: str, port: int) -> str:
    """The URL a client on this machine should call; a wildcard bind is not dialable."""
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"


def _merge_otel_attrs(existing: str | None, vendor: str | None) -> str | None:
    """Set `ai.vendor` in a `k=v,k=v` string, keeping the other attributes and order."""
    if not vendor:
        return existing
    attrs = dict(
        pair.split("=", 1) for pair in (existing or "").split(",") if "=" in pair
    )
    attrs["ai.vendor"] = vendor
    return ",".join(f"{k}={v}" for k, v in attrs.items())


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        settings = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CommandError(
            f"{path} is not valid JSON ({e}); refusing to overwrite it"
        ) from e
    if not isinstance(settings, dict):
        raise CommandError(f"{path} is not a JSON object; refusing to overwrite it")
    return settings


def _is_approval_hook(hook: dict[str, Any]) -> bool:
    command = hook.get("command")
    if hook.get("type") != "command" or not isinstance(command, str):
        return False
    try:
        return shlex.split(command) in (
            [sys.executable, "-I", "-m", APPROVALS_MODULE],
            [sys.executable, "-I", "-m", "padwan_proxy.jev"],
        )
    except ValueError:
        return False


def _merge_approvals(
    settings: dict[str, Any], *, enabled: bool | None, path: Path
) -> dict[str, Any]:
    hooks = settings.get("hooks", {})
    permissions = settings.get("permissions", {})
    if not isinstance(hooks, dict):
        raise CommandError(f"{path} has non-object 'hooks'; refusing to overwrite it")
    if not isinstance(permissions, dict):
        raise CommandError(
            f"{path} has non-object 'permissions'; refusing to overwrite it"
        )

    existing = hooks.get("PreToolUse", [])
    if not isinstance(existing, list):
        raise CommandError(
            f"{path} has non-array 'hooks.PreToolUse'; refusing to overwrite it"
        )
    groups: list[dict[str, Any]] = []
    target: dict[str, Any] | None = None
    for group in existing:
        if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
            raise CommandError(
                f"{path} has invalid 'hooks.PreToolUse'; refusing to overwrite it"
            )
        commands = group["hooks"]
        if not all(isinstance(command, dict) for command in commands):
            raise CommandError(
                f"{path} has invalid 'hooks.PreToolUse'; refusing to overwrite it"
            )
        merged = {
            **group,
            "hooks": [
                {
                    **command,
                    "command": shlex.join(
                        [sys.executable, "-I", "-m", APPROVALS_MODULE]
                    ),
                }
                if enabled is None and _is_approval_hook(command)
                else command
                for command in commands
                if enabled is None or not _is_approval_hook(command)
            ],
        }
        groups.append(merged)
        if target is None and group.get("matcher") == "*":
            target = merged

    if enabled:
        if target is None:
            target = {"matcher": "*", "hooks": []}
            groups.append(target)
        target["hooks"].append(
            {
                "type": "command",
                "command": shlex.join([sys.executable, "-I", "-m", APPROVALS_MODULE]),
                "timeout": 20,
            }
        )

    merged_settings = dict(settings)
    if enabled or "PreToolUse" in hooks:
        merged_settings["hooks"] = {**hooks, "PreToolUse": groups}
    return merged_settings


def write_claude_config(
    config_dir: Path,
    *,
    host: str,
    port: int,
    model: str,
    small_model: str | None,
    backend_url: str | None,
    timeout: float,
    context_window: int | None = None,
    approvals: bool | None = None,
    approvals_url: str | None = None,
    approvals_model: str | None = None,
    approvals_confidence: float | None = None,
) -> Path:
    """Write the env this proxy owns into `config_dir/settings.json`, keeping the rest.

    `config_dir` is a CLAUDE_CONFIG_DIR: permissions, hooks and unrelated env keys
    already there survive, and only `ai.vendor` is touched inside
    OTEL_RESOURCE_ATTRIBUTES.
    """
    path = config_dir / SETTINGS_FILE
    settings = _load_settings(path)
    env = settings.get("env", {})
    if not isinstance(env, dict):
        raise CommandError(f"{path} has a non-object 'env'; refusing to overwrite it")

    owned = json.loads(
        _env.get_template("claude_settings.json.j2").render(
            base_url=client_base_url(host, port),
            model=model,
            small_model=small_model or model,
            timeout_ms=str(int(timeout * 1000)),
            context_window=str(context_window) if context_window else None,
            otel_attrs=_merge_otel_attrs(
                env.get("OTEL_RESOURCE_ATTRIBUTES"), backend_vendor(backend_url)
            ),
        )
    )
    if approvals is True:
        owned["CLAUDE_CODE_AUTO_MODE_SERVER"] = "0"
        owned["PADWAN_PROXY_APPROVALS_LOG"] = str(
            config_dir.resolve() / "approvals.jsonl"
        )
        if env_file := os.environ.get("PADWAN_PROXY_APPROVALS_ENV_FILE"):
            owned["PADWAN_PROXY_APPROVALS_ENV_FILE"] = str(
                Path(env_file).expanduser().resolve()
            )
        if approvals_url:
            owned["PADWAN_PROXY_APPROVALS_URL"] = approvals_url
        if approvals_model:
            owned["PADWAN_PROXY_APPROVALS_MODEL"] = approvals_model
        if approvals_confidence is not None:
            owned["PADWAN_PROXY_APPROVALS_MIN_CONFIDENCE"] = str(approvals_confidence)
    settings = _merge_approvals(settings, enabled=approvals, path=path)

    merged_env = {**env, **owned}
    for key in list(merged_env):
        if key.startswith("PADWAN_PROXY_JEV_"):
            value = merged_env.pop(key)
            if approvals is None:
                merged_env.setdefault(
                    key.replace("PADWAN_PROXY_JEV_", "PADWAN_PROXY_APPROVALS_", 1),
                    value,
                )
    if approvals is not None:
        # Stale keys would aim the hook at the previous backend: a URL at a proxy no
        # longer serving a local model, a gate or model name at the wrong one.
        for key in (
            "PADWAN_PROXY_APPROVALS_URL",
            "PADWAN_PROXY_APPROVALS_MODEL",
            "PADWAN_PROXY_APPROVALS_MIN_CONFIDENCE",
        ):
            if key not in owned:
                merged_env.pop(key, None)

    config_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**settings, "env": merged_env}, indent=2) + "\n")
    return path
