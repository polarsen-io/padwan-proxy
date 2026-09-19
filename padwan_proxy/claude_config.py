from __future__ import annotations

import json
from pathlib import Path
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

    config_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**settings, "env": {**env, **owned}}, indent=2) + "\n")
    return path
