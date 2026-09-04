# Requires BuildKit (docker buildx build) for the git ADD below.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS build

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# gravier is a local editable dependency (../../gravier) until published;
# clone it where the lockfile expects it
ADD https://github.com/Andarius/gravier.git /src/gravier

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
