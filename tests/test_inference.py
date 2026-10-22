"""Проверки загрузки конкретной версии и границ выделенных фрагментов"""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.config import get_settings
from src.services.inference import load_inference, positions_to_spans


def test_spans_preserve_unicode_and_original_text():
    """Повторы и порядок позиций не меняют результат, emoji считается одним символом"""
    spans = positions_to_spans("я😊  IDIOT\nfoo", [8, 4, 5, 6, 7, 4, 10, 11, 12])
    assert [span.model_dump() for span in spans] == [
        {"start": 4, "end": 9, "text": "IDIOT"},
        {"start": 10, "end": 13, "text": "foo"},
    ]
    assert positions_to_spans("nothing", []) == []


@pytest.mark.parametrize("positions", [[-1], [3], [True], [1.5], ["1"]])
def test_spans_reject_invalid_model_output(positions):
    """Ошибочные позиции не превращаются в неверные фрагменты ответа"""
    with pytest.raises(ValueError, match="недопустимые позиции"):
        positions_to_spans("abc", positions)


def test_load_resolves_alias_once_and_pins_version(monkeypatch):
    """Загрузка по номеру версии защищает от смены alias между запросами к MLflow"""
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        status="READY", version="7", run_id="training-run"
    )
    model = Mock()
    model.predict.return_value = pd.DataFrame({"positions": [[]]})
    constructor = Mock(return_value=client)
    load = Mock(return_value=model)
    monkeypatch.setattr("src.services.inference.mlflow.MlflowClient", constructor)
    monkeypatch.setattr("src.services.inference.mlflow.pyfunc.load_model", load)
    monkeypatch.setattr("src.services.inference.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("src.services.inference.mlflow.set_registry_uri", Mock())
    settings = get_settings()
    service = load_inference(settings)
    client.get_model_version_by_alias.assert_called_once_with(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )
    load.assert_called_once_with(f"models:/{settings.mlflow_model_name}/7")
    client.get_model_version_by_alias.side_effect = ConnectionError("MLflow is offline")
    assert service.process("hello").model.version == "7"
    assert service.process("world").model.run_id == "training-run"
    assert client.get_model_version_by_alias.call_count == 1
    assert load.call_count == 1


def test_unready_model_is_not_loaded(monkeypatch):
    """Неготовая версия останавливает запуск приложения"""
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(status="PENDING_REGISTRATION")
    monkeypatch.setattr("src.services.inference.mlflow.MlflowClient", Mock(return_value=client))
    monkeypatch.setattr("src.services.inference.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("src.services.inference.mlflow.set_registry_uri", Mock())
    with pytest.raises(ValueError, match="не готова"):
        load_inference(get_settings())


def test_inference_rejects_wrong_number_of_predictions(fake_inference):
    """Лишние строки модели не игнорируются при выдаче ответа"""
    service = fake_inference
    service.model.predict.side_effect = None
    service.model.predict.return_value = pd.DataFrame({"positions": [[], []]})
    with pytest.raises(ValueError, match="один результат"):
        service.process("hello")
