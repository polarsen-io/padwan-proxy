---
icon: simple/claude
---

# Claude Code

## Write the client config

`--claude-config DIR` writes this backend's settings into `DIR/settings.json`, a Claude Code
config directory:

```bash
padwan-proxy --backend-url https://api.example.com/v1/ \
  -m my-model --small-model my-small-model \
  --claude-config ~/.claude-mybackend

CLAUDE_CONFIG_DIR=~/.claude-mybackend claude
```

The client then declares the real backend model names instead of Anthropic tier names, so
its own telemetry is labelled with the model that served the request.

- Existing keys (permissions, hooks, unrelated env) are preserved. Only the env this proxy
  owns is rewritten, and inside `OTEL_RESOURCE_ATTRIBUTES` only `ai.vendor`.
- `--context-window N` is emitted as `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
- From Docker, mount the directory at the same path; see [Docker](docker.md#write-the-client-config).

## Tool search

!!! warning "Experimental"
    Deferred tools need a `padwan-ai` build with deferred-tool support, which no
    `padwan-ai` release includes yet. Install a local checkout as described in
    [Development](development.md#test-a-local-padwan-ai-checkout).

Enable [tool search](https://code.claude.com/docs/en/mcp#configure-tool-search) explicitly
for a custom backend:

```bash
ENABLE_TOOL_SEARCH=true \
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
```

The proxy forwards non-deferred tools plus the schemas of tools selected by ToolSearch,
previous calls, or an explicit tool choice. Unselected deferred tools are omitted, which
reduces the tool-schema context sent to the backend. Selection uses only the current
request: after compaction removes all references and calls, the model must search again.

Do not use `--bare` (Claude Code 2.1.263 disables ToolSearch there), and leave
`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` unset.
