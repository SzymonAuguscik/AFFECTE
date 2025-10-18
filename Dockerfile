FROM python:3.12-slim AS tools
# no gcc in python:3.12-slim
RUN apt-get update && \
    apt-get install -y gcc=4:14.2.0-1 --no-install-recommends

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin

FROM python:3.12-slim AS libraries

COPY --from=tools /bin/gcc /bin/uv /bin/

WORKDIR /usr
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache \
    uv pip install -r requirements.txt --system

FROM python:3.12-slim AS app

COPY --from=libraries /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /affecte

COPY affecte.py .
COPY src ./src
