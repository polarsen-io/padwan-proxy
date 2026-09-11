import os

# env round-trip between the CLI process and granian worker processes;
# values are read at import, so import only from the RSGI target, after
# proxy_command has written the snapshot.
ENV_PREFIX = "PADWAN_PROXY_"

BACKEND_URL = os.environ.get(ENV_PREFIX + "BACKEND_URL")
MODEL = os.environ.get(ENV_PREFIX + "MODEL", "")
SMALL_MODEL = os.environ.get(ENV_PREFIX + "SMALL_MODEL")
VISION_MODEL = os.environ.get(ENV_PREFIX + "VISION_MODEL")
API_KEY_ENV = os.environ.get(ENV_PREFIX + "API_KEY_ENV")
MAX_OUTPUT_TOKENS = int(os.environ.get(ENV_PREFIX + "MAX_OUTPUT_TOKENS", "16384"))
DEFAULT_TIMEOUT = 3600.0
TIMEOUT = float(os.environ.get(ENV_PREFIX + "TIMEOUT", str(DEFAULT_TIMEOUT)))
DEFAULT_STREAM_RETRIES = 1
STREAM_RETRIES = int(
    os.environ.get(ENV_PREFIX + "STREAM_RETRIES", str(DEFAULT_STREAM_RETRIES))
)
TRACE = bool(os.environ.get(ENV_PREFIX + "TRACE"))
TRACE_CONTENT = bool(os.environ.get(ENV_PREFIX + "TRACE_CONTENT"))
VERBOSE = bool(os.environ.get(ENV_PREFIX + "VERBOSE"))
TIMINGS = bool(os.environ.get(ENV_PREFIX + "TIMINGS"))
BREAKDOWN = bool(os.environ.get(ENV_PREFIX + "BREAKDOWN"))
