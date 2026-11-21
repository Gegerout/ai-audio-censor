"""Обучение трёх моделей и сравнение результатов в MLflow"""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import Array
from sklearn.exceptions import ConvergenceWarning

from ml.data import (
    DATASET_NAME,
    DATASET_REVISION,
    download_split,
    prepare_train,
    read_samples,
    source_url,
)
from ml.eda import plt, save_dataset
from ml.evaluation import evaluate, split_train
from ml.model import SpanModel

VARIANTS = {
    "dictionary": "Словарный baseline",
    "word_logreg": "Логистическая регрессия по словам",
    "context_logreg": "Логистическая регрессия с контекстом",
}


def save_diagnostics(rows: list[dict], matrix: list[list[int]], directory: Path) -> None:
    """Сохранить предсказания, примеры ошибок и графики качества"""
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    errors = sorted((row for row in rows if row["f1"] < 1), key=lambda row: row["f1"])[:20]
    (directory / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = np.array(matrix)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    labels = ["Вне выделенных\nфрагментов", "Внутри выделенных\nфрагментов"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Предсказание модели",
        ylabel="Разметка датасета",
        title="Матрица ошибок по позициям в тексте",
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n{shares[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if shares[i, j] > 0.5 else "black",
            )
    fig.savefig(directory / "confusion_matrix.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.hist([row["f1"] for row in rows], bins=np.linspace(0, 1, 21), color="#3274a1")
    ax.set(xlabel="F1", ylabel="Количество текстов", title="Распределение F1 по текстам")
    fig.savefig(directory / "f1_distribution.png")
    plt.close(fig)


def save_comparison(results: list[dict], directory: Path) -> None:
    """Сохранить общую таблицу и вывод по валидационной выборке"""
    table = pd.DataFrame(results).sort_values("val_f1", ascending=False, kind="stable")
    table.to_csv(directory / "comparison.csv", index=False)
    best = table.iloc[0]
    lines = [
        "# Сравнение моделей",
        "",
        "Все модели обучены на одинаковых данных и проверены на общей валидационной выборке.",
        "Тестовая выборка при сравнении не использовалась.",
        "",
        "| Модель | F1 | Precision | Recall | F1 на текстах с разметкой |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {VARIANTS[row['variant']]} | {row['val_f1']:.4f} | "
            f"{row['val_precision']:.4f} | {row['val_recall']:.4f} | "
            f"{row['val_f1_annotated']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Лучший результат по основной метрике показала модель «{VARIANTS[best['variant']]}».",
            f"Её F1 на валидационной выборке равна {best['val_f1']:.4f}.",
            f"Идентификатор запуска: `{best['run_id']}`.",
            "",
            "Словарный baseline запоминает слова, которые встретились в обучающей части хотя бы",
            "дважды и не менее чем в половине случаев входили в выделенные фрагменты.",
            "Первая регрессия использует само слово. Вторая дополнительно учитывает соседние",
            "слова, а также начала и окончания длиной от двух до четырёх символов.",
            "У обеих регрессий одинаковые параметры и балансировка весов классов.",
            "",
            "F1, precision и recall рассчитаны по позициям для каждого текста и усреднены.",
            "Если обе разметки пустые, все три оценки равны 1.",
            "Если пуста только одна, все три оценки равны 0.",
            "Дополнительно сохранены F1 на текстах с непустой разметкой и micro F1,",
            "которая считается по суммарным совпадениям и ошибкам во всей выборке.",
            "",
            "Модели выделяют слова целиком и не выделяют пробелы или пунктуацию.",
            "Поэтому границы предсказаний могут отличаться от ручной разметки.",
            "Пустая разметка оценивается как отсутствие выделенных фрагментов,",
            "но не доказывает нейтральность текста. Это ограничение датасета.",
            "Результаты относятся к английским комментариям из валидационной выборки.",
            "Окончательное качество выбранной модели нужно проверить на тестовой выборке.",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Подготовить общие выборки и сохранить отдельный запуск для каждого варианта"""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Сравнение моделей выделения токсичных фрагментов")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI", f"http://localhost:{os.getenv('MLFLOW_PORT', '5001')}"
        ),
    )
    parser.add_argument("--experiment", default="toxic-spans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/training"))
    args = parser.parse_args()
    batch_id = uuid4().hex[:12]
    output = args.output_dir / batch_id
    output.mkdir(parents=True)
    raw, test = (read_samples(download_split(split, args.data_dir)) for split in ("train", "test"))
    prepared, audit = prepare_train(raw, test)
    train, validation = split_train(prepared, args.seed)
    datasets = []
    manifest = {"source": source_url("train"), "audit": audit, "seed": args.seed, "datasets": {}}
    for name, samples in (
        ("train_prepared", prepared),
        ("train_fit", train),
        ("validation", validation),
    ):
        frame, snapshot_hash, table_hash = save_dataset(samples, name, output)
        manifest["datasets"][name] = {
            "digest": table_hash[:32],
            "table_sha256": table_hash,
            "snapshot_sha256": snapshot_hash,
            "rows": len(samples),
        }
        if name != "train_prepared":
            dataset = mlflow.data.from_pandas(
                frame,
                source=source_url("train"),
                name=f"toxic-spans/{name}",
                targets="positions",
                digest=table_hash[:32],
            )
            datasets.append((dataset, "training" if name == "train_fit" else "validation"))
    manifest["lineage"] = {
        "parent": "train_prepared",
        "children": ["train_fit", "validation"],
        "operations": ["stratified_split_by_nonempty_annotation"],
        "validation_fraction": 0.2,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    results = []
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "text")]),
        outputs=Schema([ColSpec(Array("long"), "positions")]),
    )
    requirements = [
        f"{package}=={version(package)}"
        for package in (
            "mlflow",
            "pandas",
            "numpy",
            "scipy",
            "scikit-learn",
            "cloudpickle",
            "httpx",
        )
    ]
    for variant in VARIANTS:
        with mlflow.start_run(
            run_name=variant, tags={"stage": "training", "batch_id": batch_id}
        ) as run:
            run_id = run.info.run_id
            directory = output / variant
            directory.mkdir()
            mlflow.log_params(
                {
                    "dataset": DATASET_NAME,
                    "revision": DATASET_REVISION,
                    "variant": variant,
                    "seed": args.seed,
                    "validation_fraction": 0.2,
                    "threshold": 0.5,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "token_pattern": r"\b\w+\b",
                    "positive_word_rule": "any_annotated_character",
                    "metric_averaging": "per_text_macro",
                    "empty_both_score": 1,
                }
            )
            mlflow.log_params(
                {"min_occurrences": 2}
                if variant == "dictionary"
                else {
                    "C": 1.0,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 500,
                    "context": variant == "context_logreg",
                }
            )
            for dataset, context in datasets:
                mlflow.log_input(dataset, context=context)
            mlflow.log_artifact(str(manifest_path), "data")
            for path in output.glob("*.jsonl"):
                mlflow.log_artifact(str(path), "data")
            for path in [*Path("ml").glob("*.py"), Path("pyproject.toml"), Path("uv.lock")]:
                mlflow.log_artifact(str(path), "code")
            model = SpanModel(variant, args.seed)
            started = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train)
            fit_seconds = perf_counter() - started
            predictions = model.predict_positions([sample["text"] for sample in validation])
            metrics, rows, matrix = evaluate(validation, predictions)
            mlflow.log_metrics({**metrics, "fit_seconds": fit_seconds})
            save_diagnostics(rows, matrix, directory)
            mlflow.log_artifacts(str(directory), "validation")
            example = pd.DataFrame({"text": [sample["text"] for sample in validation[:3]]})
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=model,
                code_paths=["ml"],
                signature=signature,
                input_example=example,
                pip_requirements=requirements,
            )
            loaded = mlflow.pyfunc.load_model(info.model_uri)
            if (
                loaded.predict(example)["positions"].tolist()
                != model.predict(example)["positions"].tolist()
            ):
                raise ValueError("Предсказания сохранённой модели отличаются от исходных")
            mlflow.set_tag("model_reload_verified", "true")
            results.append(
                {"variant": variant, "run_id": run_id, "model_uri": info.model_uri, **metrics}
            )
            print(f"{variant}: F1={metrics['val_f1']:.4f}, run={run_id}", flush=True)
    save_comparison(results, output)
    client = mlflow.MlflowClient()
    best = max(results, key=lambda row: row["val_f1"])
    for result in results:
        client.set_tag(result["run_id"], "best_in_batch", str(result == best).lower())
        for name in ("comparison.csv", "report.md"):
            client.log_artifact(result["run_id"], str(output / name), "comparison")
    print(f"Отчёт: {output / 'report.md'}")
    print(f"Лучший запуск: {best['run_id']}")


if __name__ == "__main__":
    main()
