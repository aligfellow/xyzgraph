# Run all checks
check: lint type test

lint:
    uv run ruff format .
    uv run ruff check .

type:
    uv run ty check 

test:
    uv run pytest --cov --cov-report=xml

fix:
    uv run ruff format .
    uv run ruff check --fix .

# Build a release.
build:
    uv build