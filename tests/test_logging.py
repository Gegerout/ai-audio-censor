"""Тесты JSON форматтера логов"""

import asyncio
import json
import logging
import sys

from src.logging_config import JsonFormatter, task_context


def test_json_formatter_puts_extra_fields_into_json() -> None:
    """Поля из extra становятся ключами JSON, служебные поля LogRecord не попадают"""
    record = logging.makeLogRecord(
        {
            "name": "src.test",
            "levelname": "INFO",
            "msg": "user %s",
            "args": ("created",),
            "status": 201,
        }
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "src.test"
    assert payload["msg"] == "user created"
    assert payload["status"] == 201
    assert payload["request_id"] == "-"
    assert "args" not in payload


def test_json_formatter_keeps_traceback_on_one_line() -> None:
    """Traceback попадает в JSON, а запись остаётся одной строкой"""
    try:
        raise ValueError("broken")
    except ValueError:
        record = logging.makeLogRecord({"msg": "failed", "exc_info": sys.exc_info()})

    line = JsonFormatter().format(record)

    assert "\n" not in line
    assert "ValueError: broken" in json.loads(line)["exc_info"]


async def test_task_logs_are_isolated_and_hide_private_data(monkeypatch) -> None:
    """Параллельные задачи не смешиваются, пароль и входные данные скрыты"""
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "test")
    formatter = JsonFormatter("worker")
    record = logging.makeLogRecord({"msg": "amqp://user:p%40ss@broker"})

    async def fail(task_id):
        with task_context(task_id):
            await asyncio.sleep(0)
            try:
                raise ValueError(task_id + "-private")
            except ValueError:
                failure = logging.makeLogRecord({"msg": record.msg, "exc_info": sys.exc_info()})
                return json.loads(formatter.format(failure))

    first, second = await asyncio.gather(fail("first"), fail("second"))

    assert [first["task_id"], second["task_id"]] == ["first", "second"]
    for event in (first, second):
        assert event["env"] == "test"
        assert event["msg"] == "amqp://user:***@broker"
        assert event["exc_info"].endswith("ValueError")
        assert event["task_id"] + "-private" not in event["exc_info"]
    assert "task_id" not in json.loads(formatter.format(record))
