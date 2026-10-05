# Routing

Each request goes to one backend model:

| Request | Model |
|---|---|
| Haiku-tier model name | `--small-model` |
| Carries an image (anywhere in the history) | `--vision-model` |
| Anything else | `-m/--model` |

A conversation keeps routing to `--vision-model` while an image stays in its history. Set it
when the main model is text-only.

`--vision-mode caption` instead has `--vision-model` describe each image as text and keeps
the main model. Use it for vision models without tool calling (e.g. `pixtral-12b-2409`).
Captions are cached per image.

## Gemini

Models whose name starts with `gemini` are served through `padwan-ai`'s native Gemini client
(the `generateContent` / `streamGenerateContent` REST API), not the OpenAI-compatible shim:

```bash
export PADWAN_API_KEY=<YOUR_GEMINI_KEY>
uvx padwan-proxy --backend-url https://generativelanguage.googleapis.com/v1beta/ \
  -m gemini-2.5-flash --small-model gemini-2.5-flash-lite
```

Routing, `--claude-config`, tracing and the stream-retry window work as for OpenAI backends.
Thinking blocks are dropped on replay (their signatures only make sense to the Anthropic
API); Gemini reasoning (`thought: true` parts) surfaces as Anthropic `thinking` blocks.
