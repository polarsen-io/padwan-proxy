---
icon: lucide/shield-check
---

# Approval hook

!!! warning "Experimental"
    This is an experimental policy, not a measured safety guarantee. Use native permission
    rules for authority.

`--approvals` installs a Claude Code `PreToolUse` hook that asks a System One model whether
to allow, deny or ask for each tool call. It gives GLM and other backends custom tool
approvals; it does not replace the model inside Anthropic's built-in auto mode.

| Backend | Model | Verdicts | Needs |
|---|---|---|---|
| `--approvals jev` | Hosted [TypeSafe](https://docs.typesafe.ai/sdk/python) API (default `jev-latest`) | allow / deny / ask | `TYPESAFE_API_KEY`, `--claude-config`; billed to your TypeSafe account |
| `--approvals laya` | Local [Laya](laya.md) checkpoint (default `convaiinnovations/laya`) | Advisory: always `ask` | `laya` extra, initial weights download |

One hook, one model: the two are alternatives.

## Enable the hosted hook

1. Put `TYPESAFE_API_KEY` in a dotenv file and point the hook at it (absolute path):

    ```bash
    export PADWAN_PROXY_APPROVALS_ENV_FILE="$PWD/.env"
    ```

2. Generate the settings and start the proxy:

    ```bash
    uv run padwan-proxy --backend-url https://api.scaleway.ai/v1/ -m glm-5.2 \
      --claude-config ~/.claude-scaleway --approvals jev
    ```

3. In another terminal, start Claude and switch it to auto mode:

    ```bash
    CLAUDE_CONFIG_DIR=~/.claude-scaleway claude
    ```

4. Verify with `tail -f ~/.claude-scaleway/approvals.jsonl`: each tool call adds a line.

`--no-approvals` removes the hook; omitting both flags leaves an existing hook unchanged.
Restart the proxy and Claude after updating so the generated settings load.

## Behavior

- The hook evaluates calls only while Claude reports `auto` mode. Other modes get no hook
  decision and make no request; toggle auto mode to enable or pause it.
- The generated settings disable auto-mode server checks and keep your permission mode.
  Claude's internal classifier can still evaluate hook `ask` results.
- Existing deny/ask rules and other hooks still apply.
- No API key is written to settings. The hook reads only `TYPESAFE_API_KEY` from the env
  file, without executing shell code; an exported key takes precedence.
- Hosted allow/deny needs at least `--approval-confidence` (default 0.99) reported
  confidence and a valid probability distribution.

The model receives the session's text user requests, working directory, tool name and
arguments — not the full transcript.

## When the hook asks

The hook returns `ask` on classifier errors, a missing key, missing or compacted context,
and its internal 10-second deadline. It also asks rather than truncating authorization when
the context exceeds 16,000 user-request characters, 64 KiB of hook input, or an 8 MiB
transcript. Only text user requests are supported.

If the hook process cannot start or hits Claude's outer timeout, Claude's normal permission
handling applies. Noninteractive runs need a permission host to handle prompts.

## Approval log

Each hook writes a JSON line to `approvals.jsonl` in the config directory and to stderr. No
prompts, tool arguments or credentials are logged.

| Field | Meaning |
|---|---|
| `model` | Model that answered |
| `verdict`, `confidence` | What the model said, before the gate |
| `status` | `evaluated`, `advisory` (local verdict that cannot allow or deny), `inactive_mode`, `timeout`, or an error: `invalid_input`, `user_input_required`, `context_error`, `key_file_error`, `missing_api_key`, `client_error`, `api_error`, `invalid_response` |
| `api_called` | Whether the model was reached |
| `api_status` | HTTP status of an error answer: `404` proxy not serving `/systemone`, `413` state past Laya's window, `null` never reached |
| `api_ms` | Request time, including retries |
| `hook_ms` | Context loading and evaluation, excluding Python startup |

Invalid responses also log the approval answer (up to 2 KiB) and the validation error. These
logs have no ground-truth labels and cannot calibrate a threshold.
