import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_worker_reads_configuration_after_cli_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from unittest.mock import Mock, patch
from padwan_proxy import proxy

os.environ.update(
    PADWAN_PROXY_BACKEND_URL="https://worker.example/v1/",
    PADWAN_PROXY_MODEL="worker-main",
    PADWAN_PROXY_SMALL_MODEL="worker-small",
    PADWAN_PROXY_API_KEY_ENV="WORKER_KEY",
    PADWAN_PROXY_TIMEOUT="12.5",
)
pid = os.fork()
if pid == 0:
    with patch.object(proxy, "_make_client", return_value=Mock()) as client:
        with patch.object(proxy, "build_router") as router:
            import padwan_proxy.rsgi
            client.assert_called_once_with(
                "https://worker.example/v1/", "worker-main", "WORKER_KEY", timeout=12.5
            )
            assert router.call_args.kwargs["model"] == "worker-main"
            assert router.call_args.kwargs["small_model"] == "worker-small"
    os._exit(0)
_, status = os.waitpid(pid, 0)
raise SystemExit(os.waitstatus_to_exitcode(status))
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
