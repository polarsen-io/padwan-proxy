# Tracing

`--trace` instruments proxied requests with `padwan-ai`'s OpenTelemetry GenAI telemetry. It
needs the `trace` extra:

```bash
uvx 'padwan-proxy[trace]' --backend-url https://api.example.com/v1/ -m my-model --trace
```

| Environment | Export |
|---|---|
| `LANGFUSE_PUBLIC_KEY` set | Langfuse |
| Langfuse + `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Langfuse and OTLP |
| No Langfuse credentials | OTLP traces and metrics |

With Langfuse, requests carrying a Claude Code session id in `metadata.user_id` are grouped
into one Langfuse session.

`--trace-content` also records prompts and completions on the spans. A Claude Code turn ships
its whole context (system prompt, tool schemas, history), so spans get large.

## Request log

| Flag | Logs |
|---|---|
| `-v/--verbose` | One line per request: models, tokens, duration, output tok/s, tools a `tool_use` turn asked for |
| `-vv/--timings` | Plus the timing split: backend wait, request-translation time, proxy overhead |
| `--breakdown` | Plus a prompt split: system, tool schemas (by MCP server, heaviest first), message history |
| `--rich` | Colours and aligned columns; with `--breakdown`, a tree of every tool source |

Lines are prefixed with the Claude Code session id when the request carries one.
