import os

from .defaults import DEFAULT_STREAM_RETRIES, DEFAULT_TIMEOUT, ENV_PREFIX

# Import only in workers, after the CLI writes its environment snapshot.
BACKEND_URL = os.environ.get(ENV_PREFIX + "BACKEND_URL")
MODEL = os.environ.get(ENV_PREFIX + "MODEL", "")
SMALL_MODEL = os.environ.get(ENV_PREFIX + "SMALL_MODEL")
VISION_MODEL = os.environ.get(ENV_PREFIX + "VISION_MODEL")
VISION_MODE = os.environ.get(ENV_PREFIX + "VISION_MODE", "route")
API_KEY_ENV = os.environ.get(ENV_PREFIX + "API_KEY_ENV")
MAX_OUTPUT_TOKENS = int(os.environ.get(ENV_PREFIX + "MAX_OUTPUT_TOKENS", "16384"))
TIMEOUT = float(os.environ.get(ENV_PREFIX + "TIMEOUT", str(DEFAULT_TIMEOUT)))
STREAM_RETRIES = int(
    os.environ.get(ENV_PREFIX + "STREAM_RETRIES", str(DEFAULT_STREAM_RETRIES))
)
TRACE = bool(os.environ.get(ENV_PREFIX + "TRACE"))
TRACE_CONTENT = bool(os.environ.get(ENV_PREFIX + "TRACE_CONTENT"))
VERBOSE = bool(os.environ.get(ENV_PREFIX + "VERBOSE"))
TIMINGS = bool(os.environ.get(ENV_PREFIX + "TIMINGS"))
BREAKDOWN = bool(os.environ.get(ENV_PREFIX + "BREAKDOWN"))
RICH = bool(os.environ.get(ENV_PREFIX + "RICH"))
LAYA_MODEL = os.environ.get(ENV_PREFIX + "LAYA_MODEL")
LAYA_SUBFOLDER = os.environ.get(ENV_PREFIX + "LAYA_SUBFOLDER") or None
