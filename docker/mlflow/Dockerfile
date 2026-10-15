ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Серверу нужны только зависимости группы tracking
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group tracking


FROM ${PYTHON_IMAGE}

RUN useradd --system --create-home --shell /usr/sbin/nologin mlflow \
    && mkdir /mlartifacts \
    && chown mlflow:mlflow /mlartifacts

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]

# Клиенты передают артефакты через сервер, доступ к volume им не нужен
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--serve-artifacts", "--artifacts-destination", "/mlartifacts"]
