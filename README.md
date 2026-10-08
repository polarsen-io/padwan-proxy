<p align="center">
  <img src="https://raw.githubusercontent.com/polarsen-io/padwan-proxy/master/docs/static/logo-hood.png" alt="Padwan Proxy" width="120">
</p>

# Padwan Proxy

[![CI](https://github.com/polarsen-io/padwan-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/polarsen-io/padwan-proxy/actions/workflows/ci.yml)
[![Docker](https://github.com/polarsen-io/padwan-proxy/actions/workflows/docker.yml/badge.svg)](https://github.com/polarsen-io/padwan-proxy/actions/workflows/docker.yml)
[![Release](https://img.shields.io/github/v/release/polarsen-io/padwan-proxy)](https://github.com/polarsen-io/padwan-proxy/releases)
[![Docs](https://img.shields.io/badge/docs-polarsen--io.github.io-1A8B9E)](https://polarsen-io.github.io/padwan-proxy)

Serve the Anthropic Messages API (`/v1/messages` + `count_tokens`) on top of any OpenAI-compatible backend, so Anthropic clients such as Claude Code can use it. Built on [`padwan-ai`](https://github.com/polarsen-io/padwan-ai)'s Anthropic↔OpenAI translation layer, served by [granian](https://github.com/emmett-framework/granian) via [gravier](https://github.com/Andarius/gravier).

![Claude Code running GLM 5.2 on Scaleway through padwan-proxy](https://raw.githubusercontent.com/polarsen-io/padwan-proxy/master/docs/static/demo.gif)

```bash
export PADWAN_API_KEY=<YOUR_BACKEND_KEY>
uvx padwan-proxy --backend-url https://api.example.com/v1/ \
  -m my-model --small-model my-small-model \
  --claude-config ~/.claude-mybackend

CLAUDE_CONFIG_DIR=~/.claude-mybackend claude
```

Or without writing a config: `ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude`.

Or with Docker, mounting the config dir at the same path so `--claude-config` writes it on the host:

```bash
mkdir -p ~/.claude-mybackend
docker run -d --rm --name padwan-proxy -p 127.0.0.1:4000:4000 \
  --user "$(id -u):$(id -g)" -e PADWAN_API_KEY \
  -v ~/.claude-mybackend:$HOME/.claude-mybackend \
  ghcr.io/polarsen-io/padwan-proxy:latest \
  --backend-url https://api.example.com/v1/ -m my-model --small-model my-small-model \
  --claude-config $HOME/.claude-mybackend

CLAUDE_CONFIG_DIR=~/.claude-mybackend claude
```

A [`compose.yaml`](compose.yaml) does the same from a `.env`; see the Docker docs.

**Documentation: <https://polarsen-io.github.io/padwan-proxy>** — routing, Docker
(`ghcr.io/polarsen-io/padwan-proxy`), tracing, tool approvals, and the full option reference.

## Development

```bash
uv sync --group dev
just ci    # lint + type check + tests
just docs  # serve the docs locally
```
