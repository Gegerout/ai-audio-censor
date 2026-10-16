# Обе стадии строятся из одного образа, потому что .venv ссылается на его интерпретатор Python
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS builder

# Версия uv зафиксирована и совпадает с локальной, чтобы uv.lock читался одинаково
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv

# Байткод компилируется при сборке, а не при первом запуске контейнера
# Кэш uv лежит на другой файловой системе, поэтому пакеты копируются, а не связываются ссылками
# Python берётся из образа, свой uv скачивать не должен
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Зависимости отдельным слоем, он пересобирается только при изменении pyproject.toml или uv.lock
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM ${PYTHON_IMAGE}

# Пользователь без прав root, без домашней папки и без входа в систему
RUN useradd --system --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Файлы остаются за root, процесс app может их только читать и не подменит код
COPY --from=builder /app/.venv /app/.venv
# Из pyproject.toml эндпоинт /api/v1/version берёт версию приложения
COPY pyproject.toml ./
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"

USER app

EXPOSE 8000

# В slim образе нет curl, поэтому проверка написана на стандартной библиотеке Python
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

# В exec форме uvicorn становится первым процессом и сам получает SIGTERM от docker stop
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
