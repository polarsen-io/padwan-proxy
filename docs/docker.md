# Docker

Images are published to `ghcr.io/polarsen-io/padwan-proxy` with the version and `latest` on
every release, plus `edge` on each push to master.

```bash
docker run --rm -p 4000:4000 -e PADWAN_API_KEY=<YOUR_BACKEND_KEY> \
  ghcr.io/polarsen-io/padwan-proxy:latest \
  --backend-url https://api.example.com/v1/ -m my-model
```

The entrypoint binds `0.0.0.0`; everything after the image name is passed to `padwan-proxy`.
Run `--claude-config` on the host: the container has no Claude Code config to write.

## Laya variant

`just build-laya` builds an image with the [local Laya approval model](laya.md)
(`--build-arg LAYA=1`). It is several GB because the PyPI torch wheels carry CUDA. Run it on
a GPU and keep the weights in a volume:

```bash
docker run --rm --gpus all -p 4000:4000 -v laya-cache:/cache/huggingface \
  padwan-proxy:laya --backend-url https://api.example.com/v1/ -m my-model \
  --approvals laya
```

The hook runs next to Claude Code, so set `PADWAN_PROXY_APPROVALS_URL=http://127.0.0.1:4000`
in your Claude settings.
