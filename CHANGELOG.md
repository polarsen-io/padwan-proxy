## 0.1.0 (2026-10-08)

First release of padwan-proxy: the Anthropic Messages API on top of any OpenAI-compatible backend,
for Claude Code and other Anthropic clients.

- Routing to small, vision (route or caption mode) and main models
- `--claude-config` writes Claude Code settings with the real backend model names
- Stream replay before first output, per-gap read timeout, output-token cap
- Request log (`-v`, `-vv`, `--breakdown`, `--rich`) prefixed with the Claude Code session id
- Tracing to Langfuse and/or OTLP, with Claude Code sessions and optional content capture
- Experimental `--approvals jev|laya` tool-approval hook (local Laya verdicts are advisory)
- Docker image on GHCR (`latest`, version, `edge`), optional `LAYA=1` CUDA build

Docs: https://polarsen-io.github.io/padwan-proxy

