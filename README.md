# Padwan Proxy

[![CI](https://github.com/polarsen-io/padwan-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/polarsen-io/padwan-proxy/actions/workflows/ci.yml)
[![Docker](https://github.com/polarsen-io/padwan-proxy/actions/workflows/docker.yml/badge.svg)](https://github.com/polarsen-io/padwan-proxy/actions/workflows/docker.yml)
[![Release](https://img.shields.io/github/v/release/polarsen-io/padwan-proxy)](https://github.com/polarsen-io/padwan-proxy/releases)
[![ghcr.io](https://img.shields.io/badge/ghcr.io-padwan--proxy-2496ED?logo=docker&logoColor=white)](https://github.com/polarsen-io/padwan-proxy/pkgs/container/padwan-proxy)

Serve the Anthropic Messages API (`/v1/messages` + `count_tokens`) on top of any OpenAI-compatible backend, so Anthropic clients (e.g. Claude Code) can use it. Built on [`padwan-llm`](https://github.com/polarsen-io/padwan-llm)'s Anthropic↔OpenAI translation layer, served by [granian](https://github.com/emmett-framework/granian) (RSGI) via [gravier](https://github.com/Andarius/gravier).

```bash
export OPENAI_API_KEY=...  # backend key (or pass --api-key-env MY_VAR)
uvx padwan-proxy --backend-url https://api.example.com/v1/ \
  -m my-model --small-model my-small-model

# then point the client at it
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
```

## Docker

Images are published to GHCR on every release, plus `edge` on each push to master.

```bash
docker run --rm -p 4000:4000 -e PADWAN_API_KEY=... \
  ghcr.io/polarsen-io/padwan-proxy:latest \
  --backend-url https://api.example.com/v1/ -m my-model
```

The entrypoint already binds `0.0.0.0`; everything after the image name is passed
straight to `padwan-proxy`.

## Routing

- Requests for haiku-tier models go to `--small-model`.
- Requests carrying images go to `--vision-model` (set it when the main model is text-only; a conversation keeps routing there while an image stays in its history).
- Everything else goes to `-m/--model`.

## Options

- `--backend-url` — OpenAI-compatible endpoint (default: `$PADWAN_BASE_URL`).
- `--api-key-env VAR` — env var holding the backend key (default: `PADWAN_API_KEY`, then `OPENAI_API_KEY`).
- `--max-output-tokens` — cap on `max_tokens` forwarded to the backend (default 16384; Anthropic clients ask for more than many backends allow).
- `--timeout` — backend read timeout in seconds (default 3600). It applies per gap in the stream, not to the whole request: reasoning models can stay silent for minutes before their first token. The connect timeout stays at 10s, so an unreachable backend still fails fast.
- `--stream-retries` — replays of a stream that fails before any event reached the client (default 1). Nothing is replayed once the client has seen output, and permanent failures (4xx, rate limits, quota) are never retried.
- `-v/--verbose` — log each proxied request (models, tokens, duration); `-vv/--timings` adds the timing split (backend wait, request-translation time, proxy overhead).
- `--breakdown` — like `-v`, plus a second line splitting the prompt into system, tool schemas (grouped by MCP server, heaviest first) and message history, so you can see what is filling the context.
- `--trace` — instrument proxied requests with padwan-llm's OTel GenAI telemetry (Langfuse when `LANGFUSE_PUBLIC_KEY` is set, OTLP otherwise; needs the `trace` extra). With Langfuse, requests carrying a Claude Code session id in `metadata.user_id` are grouped into one Langfuse session.
- `--trace-content` — like `--trace`, plus prompts and completions recorded on the spans (a Claude Code turn ships its whole context: system prompt, tool schemas, history).
- `-p/--port` (4000), `--host` (127.0.0.1).

## Development

```bash
uv sync --group dev
just ci   # lint + type check + tests
```
