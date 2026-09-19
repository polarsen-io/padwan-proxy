import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest


@dataclass(frozen=True)
class Provider:
    """Live OpenAI-compatible reasoning backend configuration."""

    base_url: str
    model: str
    api_key_env: str


_CONFIGURED_ENV = (
    "PADWAN_PROXY_E2E_BASE_URL",
    "PADWAN_PROXY_E2E_MODEL",
)
_CONFIGURED_API_KEY_ENV = (
    "PADWAN_PROXY_E2E_API_KEY"
    if os.environ.get("PADWAN_PROXY_E2E_API_KEY")
    else "PADWAN_API_KEY"
)


def _missing(*names: str) -> bool:
    return any(not os.environ.get(name) for name in names)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


def _stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.fixture(
    params=[
        pytest.param(
            Provider(
                os.environ.get("PADWAN_PROXY_E2E_BASE_URL", ""),
                os.environ.get("PADWAN_PROXY_E2E_MODEL", ""),
                _CONFIGURED_API_KEY_ENV,
            ),
            id="configured",
            marks=pytest.mark.skipif(
                _missing(*_CONFIGURED_ENV, _CONFIGURED_API_KEY_ENV),
                reason="configured endpoint, model, and API key are required",
            ),
        ),
        pytest.param(
            Provider(
                "https://api.mistral.ai/v1/",
                "magistral-small-latest",
                "MISTRAL_API_KEY",
            ),
            id="mistral",
            marks=pytest.mark.skipif(
                _missing("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
            ),
        ),
        pytest.param(
            Provider("https://api.x.ai/v1/", "grok-3-mini", "GROK_API_KEY"),
            id="grok",
            marks=pytest.mark.skipif(
                _missing("GROK_API_KEY"), reason="GROK_API_KEY not set"
            ),
        ),
    ]
)
def live_proxy(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[str, Provider]]:
    """Run the proxy CLI against one live reasoning backend."""
    provider = cast(Provider, request.param)
    port = _free_port()
    command = [
        sys.executable,
        "-c",
        "from padwan_proxy import main; main()",
        "--backend-url",
        provider.base_url,
        "--api-key-env",
        provider.api_key_env,
        "--model",
        provider.model,
        "--port",
        str(port),
    ]
    log_path = tmp_path / "proxy.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(f"proxy exited during startup:\n{log.read()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            _stop(process)
            pytest.fail("proxy did not start within 15 seconds")

        try:
            yield f"http://127.0.0.1:{port}", provider
        finally:
            _stop(process)
