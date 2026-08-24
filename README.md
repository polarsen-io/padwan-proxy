# Padwan Proxy

Serve the Anthropic Messages API (`/v1/messages` + `count_tokens`) on top of any OpenAI-compatible backend, so Anthropic clients (e.g. Claude Code) can use it. Built on [`padwan-llm`](https://github.com/polarsen-io/padwan-llm)'s Anthropic↔OpenAI translation layer, served by [granian](https://github.com/emmett-framework/granian) (RSGI) via [gravier](https://github.com/tokobib/gravier).

```bash
export OPENAI_API_KEY=...  # backend key (or pass --api-key-env MY_VAR)
uvx padwan-proxy --backend-url https://api.example.com/v1/ \
  -m my-model --small-model my-small-model

# then point the client at it
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
```

## Routing

- Requests for haiku-tier models go to `--small-model`.
- Requests carrying images go to `--vision-model` (set it when the main model is text-only; a conversation keeps routing there while an image stays in its history).
- Everything else goes to `-m/--model`.

## Options

- `--backend-url` — OpenAI-compatible endpoint (default: `$PADWAN_BASE_URL`).
- `--api-key-env VAR` — env var holding the backend key (default: `PADWAN_API_KEY`, then `OPENAI_API_KEY`).
- `--max-output-tokens` — cap on `max_tokens` forwarded to the backend (default 16384; Anthropic clients ask for more than many backends allow).
- `-v/--verbose` — log each proxied request (models, tokens, duration); `-vv/--timings` adds the timing split (backend wait, request-translation time, proxy overhead).
- `--trace` — instrument proxied requests with padwan-llm's OTel GenAI telemetry (Langfuse when `LANGFUSE_PUBLIC_KEY` is set, OTLP otherwise; needs the `trace` extra).
- `-p/--port` (4000), `--host` (127.0.0.1).

## Development

```bash
uv sync --group dev
just ci   # lint + type check + tests
```
