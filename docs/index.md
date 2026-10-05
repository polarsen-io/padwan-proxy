# Padwan Proxy

Padwan Proxy serves the Anthropic Messages API (`/v1/messages` and `/v1/messages/count_tokens`)
on top of any OpenAI-compatible backend, so Anthropic clients such as Claude Code can use it.
It is built on [`padwan-ai`](https://github.com/polarsen-io/padwan-ai)'s Anthropic↔OpenAI
translation layer and served by [granian](https://github.com/emmett-framework/granian) (RSGI)
via [gravier](https://github.com/Andarius/gravier).

**Prerequisites:** Python 3.13+ with [uv](https://docs.astral.sh/uv/), and an API key for an
OpenAI-compatible backend that supports function calling.

## Quickstart

1. Export the backend key:

    ```bash
    export PADWAN_API_KEY=<YOUR_BACKEND_KEY>
    ```

2. Start the proxy:

    ```bash
    uvx padwan-proxy --backend-url https://api.example.com/v1/ \
      -m my-model --small-model my-small-model
    ```

3. Point the client at it:

    ```bash
    ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
    ```

Claude Code now answers with `my-model`. To keep model names in one place, let the proxy
write the client config instead — see [Claude Code](claude-code.md).

## Limitations

- The proxy does not translate Anthropic thinking budgets; reasoning must be enabled on the
  backend. Models that hide their reasoning emit no thinking blocks.
- `count_tokens` is an estimate (request size / 4), not a backend tokenizer count.
- The backend context window cannot be discovered; pass it with `--context-window`.
