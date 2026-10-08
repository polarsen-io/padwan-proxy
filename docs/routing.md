---
icon: lucide/route
---

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
