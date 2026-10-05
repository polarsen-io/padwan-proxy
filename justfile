set quiet

# List available recipes
default:
    @just --list

# Run unit tests
[group('dev')]
test *args:
    uv run pytest {{ args }}

# Evaluate cached Laya checkpoints on synthetic approval cases
[group('dev')]
benchmark-laya *args:
    uv run --extra laya python -m benchmarks.laya_approvals {{ args }}

# Run live reasoning tests with keys loaded from an env file
[group('dev')]
e2e env=".env" *args:
    uv run --env-file {{ env }} pytest tests/e2e/ -m e2e {{ args }}

# Type check
[group('dev')]
check:
    uv run pyright padwan_proxy/

# Lint + format check
[group('dev')]
lint:
    uv run ruff check padwan_proxy/ tests/
    uv run ruff format --check padwan_proxy/ tests/

# Format
[group('dev')]
fmt:
    uv run ruff format padwan_proxy/ tests/

# Fix lint issues where possible
[group('dev')]
fix:
    uv run ruff check --fix padwan_proxy/ tests/
    uv run ruff format padwan_proxy/ tests/

# Lint + type check + test
[group('dev')]
ci: lint check test

# Build the docker image
[group('docker')]
build tag='padwan-proxy:latest':
    docker buildx build --load -t {{ tag }} .

# Build the docker image with the local Laya approval model (CUDA torch, multi-GB)
[group('docker')]
build-laya tag='padwan-proxy:laya':
    docker buildx build --load --build-arg LAYA=1 -t {{ tag }} .

# Point gravier at a local checkout to hack on it
[group('dev')]
gravier-local path='../../gravier':
    uv add --editable {{ path }}

# Point gravier back at the released PyPI package
[group('dev')]
gravier-pypi:
    uv remove gravier && uv add gravier

# Bump version (commitizen — updates pyproject.toml and CHANGELOG)
[group('release')]
bump *args:
    uv run --group bump cz bump {{ args }}

# Serve docs locally with hot reload
[group('docs')]
docs:
    uv run --group docs zensical serve -f zensical.toml

# Build docs
[group('docs')]
docs-build:
    uv run --group docs zensical build -f zensical.toml
