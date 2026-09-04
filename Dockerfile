FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS build

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# gravier resolves from git (see [tool.uv.sources]), so uv needs a git client
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/polarsen/padwan-proxy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project
COPY padwan_proxy/ padwan_proxy/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim-bookworm

COPY --from=build /src/polarsen/padwan-proxy/.venv /src/polarsen/padwan-proxy/.venv
ENV PATH="/src/polarsen/padwan-proxy/.venv/bin:$PATH"

EXPOSE 4000
ENTRYPOINT ["padwan-proxy", "--host", "0.0.0.0"]
