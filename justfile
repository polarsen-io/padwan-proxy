set quiet

# List available recipes
default:
    @just --list

# Run unit tests
[group('dev')]
test *args:
    uv run pytest {{ args }}

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
    docker buildx build -t {{ tag }} .

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
