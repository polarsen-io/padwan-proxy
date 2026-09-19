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

Or let the proxy write the client config, so the model names stay in one place:

```bash
padwan-proxy --backend-url https://api.example.com/v1/ \
  -m my-model --small-model my-small-model \
  --claude-config ~/.claude-mybackend

CLAUDE_CONFIG_DIR=~/.claude-mybackend claude
```

## Claude Code tool search

With a `padwan-llm` checkout containing deferred-tool support installed (see
Development below), enable [tool search](https://code.claude.com/docs/en/mcp#configure-tool-search)
explicitly for a custom backend:

```bash
ENABLE_TOOL_SEARCH=true \
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
```

The proxy forwards non-deferred tools and the schemas of tools selected by
ToolSearch, previous calls, or an explicit tool choice. Unselected deferred tools
are omitted, reducing the tool-schema context sent to the backend. Search results
retain the tool names as text. Selection uses only the current request; after
compaction removes all references and calls, rediscovery is required.

The backend must support function calling. Do not use `--bare` for this workflow
(Claude Code 2.1.263 disables ToolSearch there), and ensure
`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` is unset. For a shell wrapper such as
`pglmclaude`, add `ENABLE_TOOL_SEARCH=true` only to that wrapper's Claude invocation.

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
- `--claude-config DIR` — write this backend's settings into `DIR/settings.json`, a Claude Code config dir (`CLAUDE_CONFIG_DIR`). The client then declares the real backend model names instead of Anthropic tier names, so its own telemetry is labelled with the model that actually served the request. Existing keys in that file (permissions, hooks, unrelated env) are preserved; only the env this proxy owns is rewritten, and inside `OTEL_RESOURCE_ATTRIBUTES` only `ai.vendor`. Host-side only — the Docker image has no client config to write.
- `--context-window N` — backend context window in tokens, emitted as `CLAUDE_CODE_AUTO_COMPACT_WINDOW` by `--claude-config`. The proxy cannot discover it from the backend.
- `-v/--verbose` — log each proxied request (models, tokens, duration); `-vv/--timings` adds the timing split (backend wait, request-translation time, proxy overhead).
- `--rich` — render the request log with colours and aligned columns (rich).
- `--breakdown` — like `-v`, plus a second line splitting the prompt into system, tool schemas (grouped by MCP server, heaviest first) and message history, so you can see what is filling the context.
- `--trace` — instrument proxied requests with padwan-llm's OTel GenAI telemetry (Langfuse when `LANGFUSE_PUBLIC_KEY` is set, OTLP otherwise; needs the `trace` extra). With Langfuse, requests carrying a Claude Code session id in `metadata.user_id` are grouped into one Langfuse session.
- `--trace-content` — like `--trace`, plus prompts and completions recorded on the spans (a Claude Code turn ships its whole context: system prompt, tool schemas, history).
- `-p/--port` (4000), `--host` (127.0.0.1).

## Development

```bash
uv sync --group dev
just ci   # lint + type check + tests
```

To test changes from a local `padwan-llm` checkout without publishing:

```bash
uv pip install --python .venv/bin/python -e ../padwan-llm
uv run --no-sync padwan-proxy --backend-url https://api.example.com/v1/ -m my-model
```

Use `uv run --no-sync` for checks while testing this override; `uv sync` restores
the dependency recorded in the lockfile.

Live reasoning tests run through a real proxy process and require either
`MISTRAL_API_KEY`, `GROK_API_KEY`, or all three of
`PADWAN_PROXY_E2E_BASE_URL`, `PADWAN_PROXY_E2E_MODEL`, and
`PADWAN_PROXY_E2E_API_KEY` (or `PADWAN_API_KEY`) in an env file. A configured model must expose
OpenAI-compatible reasoning content:

```bash
just e2e .env
```

For Scaleway GLM and local Claude Code, use an env file containing `PADWAN_API_KEY`:

```bash
PADWAN_PROXY_E2E_BASE_URL=https://api.scaleway.ai/v1/ \
PADWAN_PROXY_E2E_MODEL=glm-5.2 just e2e ../padwan-llm/.env -k configured
```

This checks streaming and complete reasoning responses, OTLP traces and metrics
sent to a local collector, and an isolated Claude Code file-read tool round trip
when the `claude` CLI is installed. The CLI test uses temporary settings and files.

The proxy does not translate Anthropic thinking budgets; reasoning must be
enabled by the backend. Models that hide their reasoning cannot emit thinking
blocks through the proxy.
