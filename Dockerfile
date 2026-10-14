FROM python:3.12-slim AS tools

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

FROM tools AS app-libraries

WORKDIR /usr

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra cu130

FROM tools AS test-libraries

WORKDIR /usr

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra cpu

FROM python:3.12-slim AS runtime

WORKDIR /affecte

ENV PATH="/usr/.venv/bin:$PATH"
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY affecte.py .
COPY src ./src

FROM runtime AS app

COPY --from=app-libraries /usr/.venv /usr/.venv

FROM runtime AS unit_tests

COPY --from=test-libraries /usr/.venv /usr/.venv
