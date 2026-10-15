.PHONY: install lint fmt test check run db up down logs mlflow mlflow-logs eda train

# Зависимости строго по uv.lock и хуки pre-commit
install:
	uv sync --locked
	uv run pre-commit install

# Линтер и проверка формата, как в CI
lint:
	uv run ruff check .
	uv run ruff format --check .

# Исправить то, что ruff умеет сам, и отформатировать код
fmt:
	uv run ruff check --fix .
	uv run ruff format .

# Тесты с покрытием, при покрытии ниже 90 процентов команда падает
test:
	uv run --locked --group ml pytest

# Линтер и тесты вместе, перед push
check: lint test

# Приложение локально с перезапуском при изменении кода, БД поднимает make db
run:
	uv run uvicorn src.app:app --reload

# Только Postgres из compose, доступен на localhost
db:
	docker compose up -d --wait db

# Весь стенд в Docker, образ пересобирается из текущего кода
up:
	docker compose up -d --build --wait

# Остановить стенд, данные Postgres в volume сохраняются
down:
	docker compose down

# Логи приложения в реальном времени
logs:
	docker compose logs -f app

# Поднять MLflow и его базу, остальные сервисы не нужны
mlflow:
	docker compose up -d --build --wait mlflow

# Логи сервера экспериментов
mlflow-logs:
	docker compose logs -f mlflow

# Подготовить датасет и сохранить анализ в работающий MLflow
eda:
	uv run --locked --group ml python -m ml.eda

# Обучить три модели и сравнить их на общей валидационной выборке
train:
	uv run --locked --group ml python -m ml.train
