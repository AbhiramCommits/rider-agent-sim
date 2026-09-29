# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependency layer cached separately from the source.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/pyproject.toml /app/pyproject.toml
COPY --from=build /app/rider_sim /app/rider_sim
COPY --from=build /app/prompts /app/prompts
COPY --from=build /app/tests/fixtures /app/tests/fixtures

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "rider_sim"]
CMD ["--help"]
