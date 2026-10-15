# AI Equalizer

[![CI/CD](https://github.com/Gegerout/ai-equalizer/actions/workflows/cd.yml/badge.svg)](https://github.com/Gegerout/ai-equalizer/actions/workflows/cd.yml)

Асинхронный сервис на FastAPI и Postgres для настройки эквалайзера с помощью ИИ. Сейчас готов каркас сервиса. Проверки живости и зависимостей, версия приложения, структурированные логи, тесты, Docker и CI/CD с публикацией образа в GitHub Container Registry.

**Стек.** Python 3.11, FastAPI, uvicorn, asyncpg, pydantic-settings. Окружение и зависимости через uv, линтер и форматтер ruff, тесты pytest с coverage, pre-commit, Docker и docker compose, GitHub Actions, образы в GHCR.

## Эндпоинты

| Метод и путь | Что делает | Коды ответа |
|---|---|---|
| `GET /healthz` | проверяет, что процесс жив, в БД не ходит | 200 |
| `GET /api/v1/version` | версия приложения из `pyproject.toml` | 200 |
| `GET /api/v1/health` | версия и время ответа каждой зависимости | 200 или 503 |

`/healthz` нужен Docker и оркестраторам. Это служебный эндпоинт инфраструктуры, а не часть версионируемого API, поэтому он живёт вне `/api/v1` и не изменится с выходом v2.

Версия приложения записана в одном месте, в `pyproject.toml`. Из него её берут `/api/v1/version` и Swagger, а CD при релизе сверяет с ней git тег. Поэтому тег образа и ответ API всегда совпадают.

Ответ `/api/v1/health`, когда Postgres работает.

```json
{"status": "ok", "components": {"postgres": {"status": "ok", "version": "17.11", "latency_ms": 16.18, "error": null}}}
```

Если Postgres недоступен, приходит код 503, статус `fail` и имя ошибки, например `ConnectionRefusedError`. Проверка ограничена таймаутом `HEALTH_TIMEOUT`, поэтому зависшая база не держит запрос. Swagger открывается на http://localhost:8000/docs.

## Запуск в Docker

Нужны Docker и make. Создайте `.env` из шаблона и впишите свой пароль вместо `change-me`.

```bash
cp .env.example .env
make up
curl localhost:8000/api/v1/health
```

`make up` собирает образ, запускает Postgres и приложение и ждёт, пока оба станут healthy. Данные Postgres лежат в volume `pgdata` и переживают `make down`.

Готовый образ можно скачать из реестра без сборки.

```bash
docker pull ghcr.io/gegerout/ai-equalizer:latest
```

## MLflow

Первая часть второй лабораторной, сервер экспериментов MLflow 3.17.1.
Добавьте переменные MLFLOW из `.env.example` в свой `.env` и задайте пароль базы MLflow.
Существующий `.env` целиком заменять не нужно.

```bash
make mlflow
```

UI открывается на http://localhost:5001. Порт можно поменять через `MLFLOW_PORT`.
Для клиента на компьютере адрес сервера `http://localhost:5001`, внутри сети compose
адрес `http://mlflow:5000`.

| Компонент | Назначение | Хранение |
|---|---|---|
| `mlflow` | Tracking Server, UI и Model Registry | обращается к двум хранилищам |
| `mlflow-db` | Backend Store, параметры, метрики, связи с датасетами и версии моделей | PostgreSQL в volume `mlflow-pgdata` |
| Artifact Store | файлы моделей, графики и отчёты | отдельный volume `mlflow-artifacts` |

Сервер принимает и отдаёт артефакты через HTTP. Клиенты используют адрес MLflow
и не монтируют его volume. Поэтому артефакты доступны и с компьютера, и из контейнеров.
В метаданных эксперимента хранится адрес вида `mlflow-artifacts:/...`.

Backend Store и Artifact Store разделены. В базе лежат описания и ссылки,
а содержимое файлов лежит в отдельном хранилище. Для локальной лабораторной
используется файловое хранилище в Docker volume. При переносе на несколько серверов
его можно заменить на S3 через настройку `--artifacts-destination`.

`make mlflow-logs` показывает логи сервера. `make down` останавливает стенд,
а эксперименты и артефакты сохраняются в volumes. Команда `docker compose down -v`
удаляет volumes вместе с данными, для обычной остановки её использовать не нужно.

Зависимости сервера вынесены в группу `tracking`, их версии закреплены в `uv.lock`.
Образ приложения по прежнему устанавливает только группу `web`.
EDA, обучение и inference будут добавлены следующими шагами.

## Локальная разработка

Нужны uv и Docker для базы. Приложение читает настройки из `.env`, а Postgres поднимается из того же compose.

```bash
cp .env.example .env
make install
make db
make run
```

| Команда | Что делает |
|---|---|
| `make install` | ставит зависимости строго по `uv.lock` и хуки pre-commit |
| `make lint` | ruff и проверка формата, как в CI |
| `make fmt` | исправляет то, что ruff умеет сам, и форматирует код |
| `make test` | тесты с покрытием, ниже 90 процентов команда падает |
| `make check` | lint и test вместе, удобно перед push |
| `make run` | приложение локально с перезапуском при изменении кода |
| `make db` | только Postgres в Docker, доступен на localhost |
| `make up`, `make down` | весь стенд в Docker |
| `make logs` | логи приложения в реальном времени |

Хуки pre-commit запускаются на каждый коммит. Они проверяют ruff и формат, пробелы в конце строк, перевод строки в конце файла, синтаксис YAML и TOML, большие файлы и случайно добавленные приватные ключи.

## Настройки

Все настройки читаются из переменных окружения или файла `.env`. Шаблон лежит в `.env.example`, а сам `.env` в git не попадает.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `POSTGRES_USER` | обязательна | пользователь Postgres |
| `POSTGRES_PASSWORD` | обязательна | пароль, хранится как `SecretStr` и не попадает в логи |
| `POSTGRES_DB` | обязательна | имя базы |
| `POSTGRES_HOST` | `localhost` | хост БД при запуске без Docker, в compose приложение само ходит на `db` |
| `POSTGRES_PORT` | `5432` | порт БД, в compose это порт Postgres на localhost |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` или `ERROR` |
| `HEALTH_TIMEOUT` | `2.0` | таймаут проверки зависимости в секундах, только больше нуля |

Без пароля приложение не стартует, а compose останавливается с понятной ошибкой. Значений по умолчанию для секретов нет специально.

## Логи

Каждая строка лога это один JSON объект. Запрос получает `request_id`, он возвращается в заголовке `X-Request-ID`, и все логи одного запроса связаны этим id. Запросы healthcheck к `/healthz` пишутся на уровне DEBUG, чтобы не забивать логи.

```json
{"ts": "2026-10-01T08:41:48.012921+00:00", "level": "INFO", "logger": "src.app", "request_id": "fb96b9b0cef340be8061f7c732a59da8", "msg": "request handled", "method": "GET", "path": "/api/v1/health", "status": 200, "duration_ms": 24.1}
```

## Структура

```
src/
├── app.py              # сборка приложения, пул БД, middleware логов
├── config.py           # настройки из окружения и версия из pyproject.toml
├── logging_config.py   # JSON логи и request_id
├── schemas.py          # контракты ответов на Pydantic
├── api/
│   ├── healthz.py      # GET /healthz
│   └── v1.py           # GET /api/v1/version и /api/v1/health
└── services/
    └── health.py       # проверка Postgres с таймаутом и замером времени
tests/                  # тесты API, сервиса, настроек и логов
Dockerfile              # сборка в две стадии, запуск не от root
docker-compose.yml      # приложение и Postgres, сеть, volume, healthcheck, лимиты
.github/workflows/      # CI и CD
```

## CI/CD

**CI** в `ci.yml` проверяет линтер, формат, остальные хуки pre-commit и тесты с порогом покрытия 90 процентов. Ошибки ruff появляются аннотациями на строках кода, таблица покрытия выводится в сводке запуска. CI запускается на pull request и на push в любые ветки, кроме `main`.

**CD** в `cd.yml` запускается на push в `main` и на теги релиза. Сначала он прогоняет тот же CI, и только если всё зелёное, собирает образ и публикует его в `ghcr.io/gegerout/ai-equalizer`.

| Событие | Теги образа |
|---|---|
| push в `main` | `main` и `sha-<коммит>` |
| тег `vX.Y.Z` | `X.Y.Z`, `X.Y`, `latest` и `sha-<коммит>` |
| pull request | ничего не публикуется |

`sha` указывает на конкретный коммит, `main` на последний коммит ветки, а semver теги и `latest` двигаются только при релизе. В образ записываются OCI метки с версией, коммитом и ссылкой на репозиторий.

### Выпуск релиза

```bash
uv version --bump patch
git commit -am "chore: release v0.1.1"
git tag v0.1.1
git push origin main v0.1.1
```

`uv version` меняет версию в `pyproject.toml` и `uv.lock`. Если тег не совпадёт с версией в `pyproject.toml`, CD упадёт до публикации образа.
