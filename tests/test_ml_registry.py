"""Проверки выбора запусков и работы реестра в отдельной локальной базе"""

from copy import deepcopy

import mlflow
import pandas as pd
import pytest
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import Array

from ml.data import Sample
from ml.model import SpanModel
from ml.register import find_batch, register_versions, set_alias, validate_runs


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Создать изолированные Runs и модели без сети и общего сервера"""
    directory = tmp_path_factory.mktemp("registry")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment("registry-tests", (directory / "artifacts").as_uri())
    train = [
        Sample(row_id=index, text=f"idiot nice {index}", positions=list(range(5)))
        for index in range(10)
    ]
    frame = pd.DataFrame({"text": ["idiot nice"], "positions": ["[0, 1, 2, 3, 4]"]})
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "text")]),
        outputs=Schema([ColSpec(Array("long"), "positions")]),
    )
    runs = []
    try:
        for variant, score in (("dictionary", 0.7), ("word_logreg", 0.5), ("context_logreg", 0.6)):
            with mlflow.start_run(
                experiment_id=experiment_id,
                tags={
                    "stage": "training",
                    "batch_id": "123456abcdef",
                    "model_reload_verified": "true",
                },
            ) as run:
                mlflow.log_param("variant", variant)
                mlflow.log_metric("val_f1", score)
                for name, context in (("train_fit", "training"), ("validation", "validation")):
                    dataset = mlflow.data.from_pandas(
                        frame, name=f"toxic-spans/{name}", digest="a" * 32
                    )
                    mlflow.log_input(dataset, context=context)
                model = SpanModel(variant)
                model.fit(train)
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=model,
                    signature=signature,
                    input_example=frame[["text"]],
                    pip_requirements=[],
                )
                run_id = run.info.run_id
            runs.append(client.get_run(run_id))
        yield client, experiment_id, runs
    finally:
        mlflow.set_tracking_uri(previous)


def test_find_batch_and_selection(registry):
    """Поиск возвращает полную группу с общей валидационной выборкой"""
    client, experiment_id, expected = registry
    actual = find_batch(client, experiment_id, None)
    assert [run.info.run_id for run in actual] == [run.info.run_id for run in expected]
    assert (
        find_batch(client, experiment_id, "123456abcdef")[0].data.params["variant"] == "dictionary"
    )
    with pytest.raises(ValueError, match="12 шестнадцатеричных"):
        find_batch(client, experiment_id, "wrong'")
    with pytest.raises(ValueError, match="три разных"):
        find_batch(client, experiment_id, "000000000000")
    empty = client.create_experiment("empty")
    with pytest.raises(ValueError, match="make train"):
        find_batch(client, empty, None)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "unfinished",
        "nan",
        "unchecked",
        "no_model",
        "no_dataset",
        "different_dataset",
    ],
)
def test_rejects_incomplete_or_incomparable_runs(registry, problem):
    """Неполные или несопоставимые эксперименты не допускаются к регистрации"""
    runs = deepcopy(registry[2])
    if problem == "missing":
        runs.pop()
    elif problem == "duplicate":
        runs[1] = runs[0]
    elif problem == "unfinished":
        runs[0].info._status = "RUNNING"
    elif problem == "nan":
        runs[0].data.metrics["val_f1"] = float("nan")
    elif problem == "unchecked":
        runs[0].data.tags["model_reload_verified"] = "false"
    elif problem == "no_model":
        runs[0].outputs.model_outputs.clear()
    elif problem == "no_dataset":
        runs[0].inputs.dataset_inputs.clear()
    else:
        runs[0].inputs.dataset_inputs[0].dataset._digest = "different"
    with pytest.raises(ValueError):
        validate_runs(runs)


def test_registration_reuses_versions_and_alias_switches(registry):
    """Повторный вызов не создаёт копий, alias выбирает нужную версию и исходный Run"""
    client, _, runs = registry
    versions = register_versions(client, "test-detector", runs)
    assert [str(item.version) for item in versions] == ["1", "2", "3"]
    assert [item.run_id for item in versions] == [run.info.run_id for run in runs]
    repeated = register_versions(client, "test-detector", runs)
    assert [str(item.version) for item in repeated] == ["1", "2", "3"]
    assert len(client.search_model_versions("name = 'test-detector'")) == 3
    first = set_alias(client, "test-detector", "champion", "1")
    assert str(first.version) == "1" and first.run_id == runs[0].info.run_id
    third = set_alias(client, "test-detector", "champion", "3")
    assert str(third.version) == "3" and third.run_id == runs[2].info.run_id
    loaded = mlflow.pyfunc.load_model("models:/test-detector@champion")
    assert loaded.unwrap_python_model().variant == "context_logreg"
    with pytest.raises(mlflow.exceptions.MlflowException):
        set_alias(client, "test-detector", "champion", "999")
    assert str(client.get_model_version_by_alias("test-detector", "champion").version) == "3"
    with pytest.raises(ValueError, match="имени модели"):
        register_versions(client, "wrong'name", runs)
