---
icon: lucide/sliders-horizontal
---

# Options

`padwan-proxy --help` is the canonical list. This page adds the behavior behind each option.

## Backend

| Option | Default | Description |
|---|---|---|
| `--backend-url` | `$PADWAN_BASE_URL` | OpenAI-compatible endpoint. |
| `--api-key-env VAR` | `PADWAN_API_KEY`, then `OPENAI_API_KEY` | Env var holding the backend key. |
| `-m/--model` | required | Model for main requests. |
| `--small-model` | — | Model for haiku-tier requests. See [Routing](routing.md). |
| `--vision-model` | — | Model for requests carrying images. |
| `--vision-mode` | `route` | `route` or `caption`. See [Routing](routing.md). |
| `--max-output-tokens` | `16384` | Cap on `max_tokens` forwarded to the backend. Anthropic clients ask for more than many backends allow. |
| `--timeout` | `3600` s | Read timeout per gap in the stream, not per request: reasoning models can stay silent for minutes before the first token. The connect timeout stays at 10 s. |
| `--stream-retries` | `1` | Replays of a stream that fails before any event reached the client. Nothing is replayed once the client has seen output; 4xx, rate-limit and quota errors are never retried. When every attempt fails before output, the client gets the backend error as an HTTP status instead of a stream. |

## Server

| Option | Default | Description |
|---|---|---|
| `-p/--port` | `4000` | Listen port. |
| `--host` | `127.0.0.1` | Bind address. |

## Claude Code

| Option | Default | Description |
|---|---|---|
| `--claude-config DIR` | — | Write settings into `DIR/settings.json`. See [Claude Code](claude-code.md). |
| `--context-window N` | — | Emitted as `CLAUDE_CODE_AUTO_COMPACT_WINDOW`. |

## Logging and tracing

See [Tracing](tracing.md) for `-v`, `-vv/--timings`, `--breakdown`, `--rich`, `--trace` and
`--trace-content`.

## Tool approvals

See [Approval hook](approvals.md) for `--approvals`, `--approval-model`,
`--approval-subfolder`, `--approval-confidence` and `--no-approvals`.
