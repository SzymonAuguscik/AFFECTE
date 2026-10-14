FROM python:3.12-slim AS tools

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

FROM tools AS libraries

WORKDIR /usr

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

FROM python:3.12-slim AS app

WORKDIR /affecte

COPY --from=libraries /usr/.venv /usr/.venv

ENV PATH="/usr/.venv/bin:$PATH"
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY affecte.py .
COPY src ./src
