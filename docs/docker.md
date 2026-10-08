---
icon: simple/docker
---

# Docker

Images are published to `ghcr.io/polarsen-io/padwan-proxy` with the version and `latest` on
every release, plus `edge` on each push to master. Releases also publish the Laya variant as
`<version>-laya` and `laya`. Both are defined in `docker-bake.hcl`.

```bash
docker run --rm -p 4000:4000 -e PADWAN_API_KEY=<YOUR_BACKEND_KEY> \
  ghcr.io/polarsen-io/padwan-proxy:latest \
  --backend-url https://api.example.com/v1/ -m my-model
```

The entrypoint binds `0.0.0.0`; everything after the image name is passed to `padwan-proxy`.

## Write the client config

To let [`--claude-config`](claude-code.md) write the host's Claude Code config, mount the
directory at the same path and run as your user, so the file stays yours:

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

The written base URL is `http://127.0.0.1:4000`, so publish the same port. Do not combine
this with `--approvals`: the hook command points at the container's Python.

## Compose

[`compose.yaml`](https://github.com/polarsen-io/padwan-proxy/blob/master/compose.yaml) runs the
same setup with the config in `~/.claude-padwan`. Put the backend settings in `.env`:

```bash title=".env"
PADWAN_API_KEY=<YOUR_BACKEND_KEY>
PADWAN_BASE_URL=https://api.example.com/v1/
PADWAN_MODEL=my-model
PADWAN_SMALL_MODEL=my-small-model  # optional, defaults to PADWAN_MODEL
PADWAN_CONTEXT_WINDOW=256000       # optional, defaults to 200000
PADWAN_PORT=4000                   # optional
```

```bash
mkdir -p ~/.claude-padwan
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d
CLAUDE_CONFIG_DIR=~/.claude-padwan claude
```

Pass any other option by extending `command`.

## Laya variant

`just build-laya` builds an image with the [local Laya approval model](laya.md)
(bake target `laya`). It is several GB because the PyPI torch wheels carry CUDA. Run it on
a GPU and keep the weights in a volume:

```bash
docker run --rm --gpus all -p 4000:4000 -v laya-cache:/cache/huggingface \
  ghcr.io/polarsen-io/padwan-proxy:laya --backend-url https://api.example.com/v1/ -m my-model \
  --approvals laya
```

The hook runs next to Claude Code, so set `PADWAN_PROXY_APPROVALS_URL=http://127.0.0.1:4000`
in your Claude settings.
