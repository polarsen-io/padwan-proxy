FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS build

# LAYA=1 adds the local approval model: CUDA torch, several GB of image.
ARG LAYA=""
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

WORKDIR /src/polarsen/padwan-proxy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project ${LAYA:+--extra laya}
COPY padwan_proxy/ padwan_proxy/
RUN uv sync --frozen --no-dev --no-editable ${LAYA:+--extra laya}

FROM python:3.14-slim-bookworm

COPY --from=build /src/polarsen/padwan-proxy/.venv /src/polarsen/padwan-proxy/.venv
# --laya downloads weights here; mount a volume to keep them across containers.
ENV PATH="/src/polarsen/padwan-proxy/.venv/bin:$PATH" HF_HOME=/cache/huggingface

EXPOSE 4000
ENTRYPOINT ["padwan-proxy", "--host", "0.0.0.0"]
