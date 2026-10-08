---
icon: lucide/code
---

# Development

```bash
uv sync --group dev
just ci   # lint + type check + tests
just docs # serve this site locally
```

## Test a local padwan-ai checkout

```bash
uv pip install --python .venv/bin/python -e ../padwan-ai
uv run --no-sync padwan-proxy --backend-url https://api.example.com/v1/ -m my-model
```

Use `uv run --no-sync` for checks while the override is installed; `uv sync` restores the
locked version.

## Demo GIF

`just demo` re-records `docs/static/demo.gif` from `docs/demo.tape` with
[vhs](https://github.com/charmbracelet/vhs). It builds the image from the checkout, starts the
[compose](docker.md#compose) stack with the `.env` backend, runs Claude Code in a throwaway
`HOME`, then tears both down. Set `PADWAN_PORT` if port 4000 is taken.

## Live tests

Live tests run through a real proxy process. They check streaming and complete reasoning
responses, OTLP traces and metrics sent to a local collector, and — when the `claude` CLI is
installed — an isolated Claude Code file-read round trip using temporary settings and files.

The env file needs one of:

- `MISTRAL_API_KEY`
- `GROK_API_KEY`
- `PADWAN_PROXY_E2E_BASE_URL`, `PADWAN_PROXY_E2E_MODEL` and `PADWAN_PROXY_E2E_API_KEY`
  (or `PADWAN_API_KEY`)

The model must expose OpenAI-compatible reasoning content.

```bash
just e2e .env
```

For Scaleway GLM with an env file containing `PADWAN_API_KEY`:

```bash
PADWAN_PROXY_E2E_BASE_URL=https://api.scaleway.ai/v1/ \
PADWAN_PROXY_E2E_MODEL=glm-5.2 just e2e .env -k configured
```
