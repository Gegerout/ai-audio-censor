.PHONY: install lint fmt test check run db up down logs mlflow mlflow-logs eda train register rabbitmq queue-limit worker worker-logs source experiment

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

# Приложение локально с перезапуском при изменении кода, БД и брокер поднимают make db и make rabbitmq
run:
	uv run uvicorn src.app:app --reload

# Только Postgres из compose, доступен на localhost
db:
	docker compose up -d --wait db

# Весь стенд в Docker, образ пересобирается из текущего кода
# Лимит очереди и очереди задаются до запуска остальных сервисов
up: rabbitmq
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

# Поднять RabbitMQ, задать лимит очереди задач и объявить очереди
rabbitmq:
	docker compose up -d --wait rabbitmq
	docker compose run --rm --build rabbitmq-setup

# Сменить лимит очереди на лету без потери сообщений, новое значение передаётся в LIMIT
queue-limit:
	docker compose run --rm -e TASKS_QUEUE_LIMIT=$(LIMIT) rabbitmq-setup

# Пересоздать воркеры, число задаёт WORKERS, prefetch и размер пула PREFETCH и CONCURRENCY
# Пустые значения не передаются, тогда берутся настройки из .env
WORKERS ?= 1
worker:
	$(if $(PREFETCH),WORKER_PREFETCH=$(PREFETCH)) $(if $(CONCURRENCY),WORKER_CONCURRENCY=$(CONCURRENCY)) \
		docker compose up -d --build --wait --no-deps --scale worker=$(WORKERS) worker

# Логи всех воркеров в реальном времени
worker-logs:
	docker compose logs -f worker

# Пакетная отправка строк файла в очередь задач, файл появляется после make eda
# RUN задаёт имя запуска и id задач, START и COUNT задают диапазон строк
SOURCE_FILE ?= $(firstword $(wildcard data/raw/*/train.csv))
RUN ?= default
source:
	docker compose run --rm --build source --file $(SOURCE_FILE) --run $(RUN) \
		$(if $(START),--start $(START)) $(if $(COUNT),--count $(COUNT))

# Эксперимент с лимитом очереди и без него, около пяти минут, данные и графики в docs/experiment
experiment:
	docker compose build source worker
	uv run --locked --group ml python -m experiments.backpressure
	uv run --locked --group ml python -m experiments.plot

# Подготовить датасет и сохранить анализ в работающий MLflow
eda:
	uv run --locked --group ml python -m ml.eda

# Обучить три модели и сравнить их на общей валидационной выборке
train:
	uv run --locked --group ml python -m ml.train

# Зарегистрировать версии моделей и оценить выбранную модель на тестовой выборке
register:
	uv run --locked --group ml python -m ml.register
