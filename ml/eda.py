"""Анализ обучающей выборки и сохранение результатов в MLflow"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv

from ml.data import (
    DATASET_NAME,
    DATASET_REVISION,
    FILE_HASHES,
    Sample,
    download_split,
    prepare_train,
    read_samples,
    source_url,
)


def save_dataset(
    samples: list[Sample], name: str, directory: Path
) -> tuple[pd.DataFrame, str, str]:
    """Сохранить снимок и вычислить хеши файла и таблицы для MLflow"""
    content = "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples)
    (directory / f"{name}.jsonl").write_text(content, encoding="utf-8")
    # Номер строки остаётся в снимке для проверки происхождения, но не становится признаком
    frame = pd.DataFrame(samples)[["text", "positions"]]
    # Строковое представление списка даёт стабильную схему целевой колонки в MLflow
    frame["positions"] = frame["positions"].map(json.dumps)
    table = frame.to_json(orient="records", force_ascii=False)
    return (
        frame,
        hashlib.sha256(content.encode()).hexdigest(),
        hashlib.sha256(table.encode()).hexdigest(),
    )


def describe_train(samples: list[Sample]) -> tuple[pd.DataFrame, Counter[str]]:
    """Измерить длины текстов и долю размеченных символов и слов"""
    rows = []
    marked_words = Counter()
    for sample in samples:
        positions = set(sample["positions"])
        words = list(re.finditer(r"\b\w+\b", sample["text"]))
        marked = [
            word.group().casefold()
            for word in words
            if any(index in positions for index in range(word.start(), word.end()))
        ]
        marked_words.update(marked)
        rows.append(
            {
                "row_id": sample["row_id"],
                "characters": len(sample["text"]),
                "words": len(words),
                "marked_characters": len(positions),
                "marked_words": len(marked),
                "marked_share": len(positions) / len(sample["text"]),
            }
        )
    return pd.DataFrame(rows), marked_words


def save_plots(stats: pd.DataFrame, words: Counter[str], directory: Path) -> None:
    """Сохранить графики длины текстов и распределения разметки"""
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.hist(stats["characters"], bins=50, color="#3274a1")
    ax.set(
        yscale="log",
        xlabel="Длина текста, символов",
        ylabel="Количество текстов, логарифмическая шкала",
    )
    ax.set_title("Распределение длины текстов")
    fig.savefig(directory / "text_lengths.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.hist(stats["marked_share"], bins=30, color="#3274a1")
    ax.set(xlabel="Доля длины текста, занятая токсичными фрагментами", ylabel="Количество текстов")
    ax.set_title("Доля текста с токсичными фрагментами")
    fig.savefig(directory / "annotation_share.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    values = [int((stats["words"] - stats["marked_words"]).sum()), int(stats["marked_words"].sum())]
    bars = ax.bar(
        ["Без токсичных фрагментов\nв разметке", "С токсичными фрагментами\nв разметке"],
        values,
        color=["#3274a1", "#c75b48"],
    )
    ax.bar_label(bars, padding=4)
    ax.set(ylabel="Количество слов", title="Распределение слов по наличию токсичных фрагментов")
    ax.margins(y=0.15)
    fig.savefig(directory / "word_balance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    common = words.most_common(15)[::-1]
    ax.barh([word for word, _ in common], [count for _, count in common], color="#c75b48")
    ax.set(xlabel="Количество вхождений", title="Наиболее частые слова с токсичными фрагментами")
    fig.savefig(directory / "marked_words.png")
    plt.close(fig)


def write_report(audit: dict[str, int], stats: pd.DataFrame, directory: Path) -> None:
    """Записать выводы анализа и ограничения разметки"""
    empty = int((stats["marked_characters"] == 0).sum())
    marked_rate = stats["marked_words"].sum() / max(stats["words"].sum(), 1)
    report = f"""# Анализ датасета Toxic Spans

Источник: [Toxic Spans на Hugging Face](https://huggingface.co/datasets/{DATASET_NAME})

## Подготовка данных

| Выборка | Количество строк |
| --- | ---: |
| Исходная обучающая | {audit["train_raw_rows"]} |
| Обучающая после очистки | {audit["train_prepared_rows"]} |
| Тестовая | {audit["test_rows"]} |

При сравнении текстов без учёта регистра и повторяющихся пробелов найдены
{audit["train_duplicate_groups"]} групп повторов.
В них входят {audit["train_duplicate_rows"]} строки.
Ещё {audit["overlap_texts"]} разных текста встречаются и в обучающей, и в тестовой выборке.
Им соответствуют {audit["overlap_rows"]} строк обучающей выборки.
Некоторые строки относятся к обеим категориям, поэтому их нельзя считать дважды.
Всего исключено {audit["removed_rows"]} строк.

Нормализация применяется только при сравнении текстов. Исходные тексты и позиции
размеченных символов сохраняются. Из каждой группы повторов удалены все строки,
чтобы не выбирать одну из возможных разметок произвольно.
Тестовая выборка сохранена без изменений. Для неё проверены целостность файла,
корректность позиций символов и наличие общих текстов с обучающей выборкой.
Для графиков и выбора признаков тестовая выборка не используется.

## Результаты анализа обучающей выборки

Медианная длина текста составляет {stats["characters"].median():.0f} символов.
У 95% текстов длина не превышает {stats["characters"].quantile(0.95):.0f} символов,
а максимальная длина составляет {stats["characters"].max()} символов.
При выборе ограничения на длину входа модели нужно учитывать длинные тексты.
На этом этапе тексты не обрезаются, чтобы сохранить все размеченные фрагменты.

В {empty} текстах не выделены токсичные фрагменты. Это {empty / len(stats):.1%} выборки.
Отсутствие выделенных фрагментов не означает, что текст нейтрален. По инструкции авторов
разметка может быть пустой, если токсичен весь текст или нельзя выделить конкретный фрагмент.
Поэтому поле toxic исходного CSV не используется как самостоятельный признак отсутствия мата.

Разметка хранит позиции символов, из которых состоят выделенные слова и фразы.
Доля текста с токсичными фрагментами рассчитывается как суммарная длина этих фрагментов
в символах, делённая на длину всего текста.

В {marked_rate:.1%} слов хотя бы часть слова входит в выделенный токсичный фрагмент.
На графиках такие слова названы словами с токсичными фрагментами.
Остальные слова не содержат выделенных символов, но это не гарантирует их нейтральность.
Слов без выделенных символов гораздо больше, поэтому accuracy не подходит
в качестве основной метрики: высокое значение можно получить, не находя токсичные фрагменты.
Для сравнения моделей выбрана F1 по позициям символов, а дополнительно будут измеряться
precision и recall. Пустые предсказания и пустую разметку нужно оценивать
по одному правилу для всех моделей.

Частотный график показывает слова, которые полностью или частично входят в токсичные фрагменты.
В разметку входят оскорбления и другие токсичные фрагменты, а не только ненормативная лексика.
Поэтому этот список нельзя считать готовым словарём мата.
Качество работы на песнях и русских текстах нужно проверять отдельно:
этот датасет содержит английские комментарии.

## Воспроизводимость и следующие эксперименты

Целостность исходных CSV проверяется по SHA256. Снимки данных, контрольные суммы
и последовательность преобразований сохранены в артефактах MLflow.
Перед обучением из очищенных данных нужно выделить валидационную выборку,
зафиксировав начальное значение генератора случайных чисел.
Признаки и словари следует строить только на обучающей части.
Варианты моделей будут сравниваться на валидационной выборке,
а тестовая выборка будет использована для итоговой оценки выбранной модели.

[Описание разметки авторами датасета](https://github.com/ipavlopoulos/toxic_spans/blob/master/ACL2022/README.md)
"""
    (directory / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Подготовить данные и записать один запуск EDA в MLflow"""
    load_dotenv()
    parser = argparse.ArgumentParser(description="EDA датасета Toxic Spans")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI", f"http://localhost:{os.getenv('MLFLOW_PORT', '5001')}"
        ),
    )
    parser.add_argument("--experiment", default="toxic-spans")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(
        run_name="eda", tags={"stage": "eda", "dataset_revision": DATASET_REVISION}
    ) as run:
        run_id = run.info.run_id
        output = args.output_dir / run_id
        output.mkdir(parents=True)
        paths = {split: download_split(split, args.data_dir) for split in FILE_HASHES}
        raw_train, test = (read_samples(paths[split]) for split in ("train", "test"))
        train, audit = prepare_train(raw_train, test)
        manifest = {"repository": DATASET_NAME, "revision": DATASET_REVISION, "datasets": {}}
        for name, samples, split, context in (
            ("train_raw", raw_train, "train", "source"),
            ("train_prepared", train, "train", "eda"),
            ("test", test, "test", "integrity_check"),
        ):
            frame, snapshot_sha256, table_sha256 = save_dataset(samples, name, output)
            dataset = mlflow.data.from_pandas(
                frame,
                source=source_url(split),
                name=f"toxic-spans/{name}",
                targets="positions",
                digest=table_sha256[:32],
            )
            mlflow.log_input(dataset, context=context)
            manifest["datasets"][name] = {
                "source": source_url(split),
                "raw_sha256": FILE_HASHES[split],
                "snapshot_sha256": snapshot_sha256,
                "table_sha256": table_sha256,
                "digest": dataset.digest,
                "rows": len(samples),
            }
        manifest["source_preparation"] = [
            "select_text_and_positions",
            "parse_and_validate_character_offsets",
            "sort_and_deduplicate_offsets",
            "preserve_source_row_id",
        ]
        manifest["lineage"] = {
            "parent": "train_raw",
            "child": "train_prepared",
            "parent_digest": manifest["datasets"]["train_raw"]["digest"],
            "child_digest": manifest["datasets"]["train_prepared"]["digest"],
            "operations": [
                "exclude_all_normalized_duplicates",
                "exclude_normalized_test_overlap",
            ],
            "normalization": "casefold and collapse whitespace for comparison only",
        }
        stats, words = describe_train(train)
        stats.to_csv(output / "train_statistics.csv", index=False)
        save_plots(stats, words, output)
        write_report(audit, stats, output)
        mlflow.log_params({"dataset": DATASET_NAME, "revision": DATASET_REVISION})
        mlflow.log_metrics(
            {
                **audit,
                "train_empty_spans": int((stats["marked_characters"] == 0).sum()),
                "train_marked_word_share": float(
                    stats["marked_words"].sum() / stats["words"].sum()
                ),
            }
        )
        mlflow.log_dict(manifest, "data/manifest.json")
        mlflow.log_dict(audit, "eda/audit.json")
        # Снимок кода позволяет повторить анализ даже до коммита текущих изменений
        for path in (
            Path(__file__),
            Path(__file__).with_name("data.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        ):
            mlflow.log_artifact(str(path), artifact_path="code")
        for path in paths.values():
            mlflow.log_artifact(str(path), artifact_path="data/raw")
        for path in output.iterdir():
            mlflow.log_artifact(
                str(path), artifact_path="data/prepared" if path.suffix == ".jsonl" else "eda"
            )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Run {run_id}")
        print(f"Отчёт {output / 'report.md'}")
        print(f"MLflow {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
