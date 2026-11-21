"""Регистрация моделей, выбор рабочей версии и итоговая оценка"""

import argparse
import math
import os
import re
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

from ml.data import download_split, read_samples, source_url
from ml.eda import save_dataset
from ml.evaluation import evaluate
from ml.train import VARIANTS, save_diagnostics


def validate_runs(runs: list[Run]) -> list[Run]:
    """Проверить полноту сравнения и совпадение выборок до регистрации"""
    if len(runs) != len(VARIANTS) or {run.data.params.get("variant") for run in runs} != set(
        VARIANTS
    ):
        raise ValueError("В группе должны быть ровно три разных варианта модели")
    reference = None
    for run in runs:
        score = run.data.metrics.get("val_f1", float("nan"))
        if run.info.status != "FINISHED" or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Нужны завершённые запуски с корректной валидационной F1")
        if run.data.tags.get("model_reload_verified") != "true":
            raise ValueError("Загрузка сохранённой модели не проверена")
        if run.outputs is None or len(run.outputs.model_outputs) != 1:
            raise ValueError("У каждого запуска должна быть одна сохранённая модель")
        datasets = sorted(
            (item.dataset.name, item.dataset.digest, item.dataset.source)
            for item in run.inputs.dataset_inputs
        )
        if {item[0] for item in datasets} != {"toxic-spans/train_fit", "toxic-spans/validation"}:
            raise ValueError("У запуска отсутствуют обучающая или валидационная выборки")
        if reference is not None and datasets != reference:
            raise ValueError("Сравнивать модели нужно на одинаковых выборках")
        reference = datasets
    return sorted(runs, key=lambda run: list(VARIANTS).index(run.data.params["variant"]))


def find_batch(client: mlflow.MlflowClient, experiment_id: str, batch_id: str | None) -> list[Run]:
    """Найти указанную группу или последнюю группу обучения"""
    if batch_id is None:
        latest = client.search_runs(
            [experiment_id],
            "tags.stage = 'training'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if not latest:
            raise ValueError("Сначала выполните make train")
        batch_id = latest[0].data.tags.get("batch_id", "")
    if not re.fullmatch(r"[0-9a-f]{12}", batch_id):
        raise ValueError("Идентификатор группы должен содержать 12 шестнадцатеричных символов")
    runs = client.search_runs(
        [experiment_id], f"tags.stage = 'training' AND tags.batch_id = '{batch_id}'", max_results=4
    )
    return validate_runs(list(runs))


def register_versions(
    client: mlflow.MlflowClient, name: str, runs: list[Run]
) -> list[ModelVersion]:
    """Создать версии с привязкой к Runs или использовать уже зарегистрированные"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(
            "В имени модели допустимы латинские буквы, цифры, точки, дефисы и подчёркивания"
        )
    runs = validate_runs(runs)
    versions = []
    for run in runs:
        model_id = run.outputs.model_outputs[0].model_id
        existing = client.search_model_versions(f"name = '{name}' AND run_id = '{run.info.run_id}'")
        matching = [
            item
            for item in existing
            if item.source == f"models:/{model_id}" and item.status == "READY"
        ]
        if matching:
            registered = min(matching, key=lambda item: int(item.version))
        else:
            registered = mlflow.register_model(
                f"models:/{model_id}",
                name,
                tags={
                    "variant": run.data.params["variant"],
                    "batch_id": run.data.tags["batch_id"],
                    "val_f1": str(run.data.metrics["val_f1"]),
                },
            )
        versions.append(registered)
    return versions


def set_alias(client: mlflow.MlflowClient, name: str, alias: str, version: str) -> ModelVersion:
    """Проверить загрузку конкретной версии и назначить ей alias"""
    selected = client.get_model_version(name, version)
    if selected.status != "READY":
        raise ValueError("Версия модели ещё не готова к использованию")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    model.predict(pd.DataFrame({"text": ["Hello", ""]}))
    client.set_registered_model_alias(name, alias, version)
    return client.get_model_version_by_alias(name, alias)


def evaluate_champion(
    client: mlflow.MlflowClient, name: str, alias: str, output_dir: Path, data_dir: Path
) -> None:
    """Зафиксировать выбранную через alias версию и оценить её на тестовых данных"""
    selected = client.get_model_version_by_alias(name, alias)
    # После разрешения alias загружаем конкретную версию, чтобы оценка не зависела от переключений
    model_uri = f"models:/{name}/{selected.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    test = read_samples(download_split("test", data_dir))
    with mlflow.start_run(
        run_name="test_champion", tags={"stage": "test", "model_name": name}
    ) as run:
        output = output_dir / run.info.run_id
        output.mkdir(parents=True)
        frame, snapshot_hash, table_hash = save_dataset(test, "test", output)
        dataset = mlflow.data.from_pandas(
            frame,
            source=source_url("test"),
            name="toxic-spans/test",
            targets="positions",
            digest=table_hash[:32],
        )
        mlflow.log_input(dataset, context="test")
        mlflow.log_params(
            {
                "model_name": name,
                "model_version": selected.version,
                "model_alias": alias,
                "training_run_id": selected.run_id,
                "model_uri": model_uri,
                "test_rows": len(test),
                "selection_metric": "val_f1",
                "metric_averaging": "per_text_macro",
                "empty_both_score": 1,
            }
        )
        predictions = model.predict(pd.DataFrame({"text": [row["text"] for row in test]}))
        metrics, rows, matrix = evaluate(test, predictions["positions"].tolist())
        metrics = {key.replace("val_", "test_", 1): value for key, value in metrics.items()}
        mlflow.log_metrics(metrics)
        save_diagnostics(rows, matrix, output)
        (output / "report.md").write_text(
            "# Итоговая оценка модели\n\n"
            f"Выбрана модель «{VARIANTS[selected.tags['variant']]}» "
            "по F1 на валидационной выборке.\n"
            f"На тестовой выборке проверено {len(test)} текстов.\n\n"
            "| Метрика | Значение |\n| --- | ---: |\n"
            + "".join(f"| {key} | {value:.4f} |\n" for key, value in metrics.items())
            + "\nТестовая выборка не использовалась для выбора модели или её параметров.\n"
            "Модель после выбора не переобучалась. Правила оценки совпадают с валидацией.\n"
            "Результаты характеризуют совпадение с разметкой английских комментариев.\n"
            "Отсутствие выделенных фрагментов не означает нейтральность текста.\n",
            encoding="utf-8",
        )
        mlflow.log_dict(
            {
                "source": source_url("test"),
                "digest": dataset.digest,
                "snapshot_sha256": snapshot_hash,
                "table_sha256": table_hash,
                "model_name": name,
                "model_version": selected.version,
                "training_run_id": selected.run_id,
            },
            "data/manifest.json",
        )
        mlflow.log_artifact(str(output / "test.jsonl"), "data")
        for path in output.iterdir():
            if path.suffix != ".jsonl" or path.name == "predictions.jsonl":
                mlflow.log_artifact(str(path), "test")
        for path in (
            Path(__file__),
            Path("ml/evaluation.py"),
            Path("ml/train.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        ):
            mlflow.log_artifact(str(path), "code")
        client.set_model_version_tag(name, selected.version, "test_run_id", run.info.run_id)
        print(f"Тестовая F1: {metrics['test_f1']:.4f}")
        print(f"Отчёт: {output / 'report.md'}")


def main() -> None:
    """Зарегистрировать группу моделей или переключить alias на существующую версию"""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Версии моделей и alias в MLflow")
    parser.add_argument("--model-name", default="toxic-spans-detector")
    parser.add_argument("--alias", default="champion")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--batch-id")
    choice.add_argument("--version")
    parser.add_argument("--experiment", default="toxic-spans")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI", f"http://localhost:{os.getenv('MLFLOW_PORT', '5001')}"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/registry"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    if args.version:
        selected = set_alias(client, args.model_name, args.alias, args.version)
    else:
        experiment = client.get_experiment_by_name(args.experiment)
        if experiment is None:
            raise ValueError("Эксперимент не найден, сначала выполните make train")
        runs = find_batch(client, experiment.experiment_id, args.batch_id)
        versions = register_versions(client, args.model_name, runs)
        best_index = max(range(len(runs)), key=lambda index: runs[index].data.metrics["val_f1"])
        selected = set_alias(client, args.model_name, args.alias, str(versions[best_index].version))
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        evaluate_champion(client, args.model_name, args.alias, args.output_dir, args.data_dir)
        for run, registered in zip(runs, versions, strict=True):
            print(f"Версия {registered.version}: {run.data.params['variant']}")
    print(f"{args.model_name}@{args.alias} → версия {selected.version}")
    print(f"Run обучения: {selected.run_id}")


if __name__ == "__main__":
    main()
