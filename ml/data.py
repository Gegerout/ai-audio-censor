"""Загрузка и проверка датасета с разметкой токсичных фрагментов"""

import ast
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import TypedDict

import httpx

DATASET_NAME = "heegyu/toxic-spans"
DATASET_REVISION = "9062568da9c6383c09193f3aa1f3c48dca925c5f"
FILE_HASHES = {
    "train": "0ae1b37c96e8adddf8c3282d158a524713c69583a240d6af9cbf03b67007178a",
    "test": "e252a2f540c5a864ca3fee4721f5d2b08b0acc1ba095d375a5d54cce9523d174",
}


class Sample(TypedDict):
    """Текст и позиции размеченных символов с номером исходной строки"""

    row_id: int
    text: str
    positions: list[int]


def source_url(split: str) -> str:
    """Вернуть адрес файла в закреплённой версии датасета"""
    if split not in FILE_HASHES:
        raise ValueError(f"Неизвестная выборка {split}")
    return f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/{DATASET_REVISION}/{split}.csv"


def download_split(split: str, directory: Path) -> Path:
    """Скачать исходный файл и проверить его контрольную сумму"""
    url = source_url(split)
    path = directory / DATASET_REVISION / f"{split}.csv"
    if path.exists():
        content = path.read_bytes()
    else:
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        content = response.content
    if hashlib.sha256(content).hexdigest() != FILE_HASHES[split]:
        raise ValueError(f"Контрольная сумма {split}.csv не совпадает с закреплённой версией")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def parse_positions(value: str, text: str) -> list[int]:
    """Проверить позиции символов и вернуть их без повторов по порядку"""
    try:
        positions = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Разметка должна содержать список позиций") from exc
    if not isinstance(positions, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(text)
        for index in positions
    ):
        raise ValueError("Разметка содержит недопустимые позиции символов")
    return sorted(set(positions))


def read_samples(path: Path) -> list[Sample]:
    """Прочитать тексты и проверить разметку каждой строки"""
    samples = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text_of_post", "position"}.issubset(reader.fieldnames or []):
            raise ValueError("В CSV отсутствуют столбцы текста или разметки")
        for row_id, row in enumerate(reader):
            text = row["text_of_post"]
            if not text or not text.strip() or row["position"] is None:
                raise ValueError(f"Пустой текст или пропущенная разметка в строке {row_id}")
            samples.append(
                Sample(row_id=row_id, text=text, positions=parse_positions(row["position"], text))
            )
    if not samples:
        raise ValueError("Датасет пуст")
    return samples


def normalize_text(text: str) -> str:
    """Свернуть регистр и пробелы только для поиска повторов"""
    return " ".join(text.casefold().split())


def prepare_train(train: list[Sample], test: list[Sample]) -> tuple[list[Sample], dict[str, int]]:
    """Убрать повторы и пересечения с тестом, сохранив исходные тексты и позиции"""
    counts = Counter(normalize_text(sample["text"]) for sample in train)
    test_texts = {normalize_text(sample["text"]) for sample in test}
    # Удаляем всю группу повторов, чтобы не выбирать произвольную разметку одного из них
    prepared = [
        sample
        for sample in train
        if counts[normalize_text(sample["text"])] == 1
        and normalize_text(sample["text"]) not in test_texts
    ]
    if not prepared:
        raise ValueError("После удаления повторов обучающая выборка пуста")
    audit = {
        "train_raw_rows": len(train),
        "test_rows": len(test),
        "train_duplicate_groups": sum(count > 1 for count in counts.values()),
        "train_duplicate_rows": sum(count for count in counts.values() if count > 1),
        "overlap_texts": len(counts.keys() & test_texts),
        "overlap_rows": sum(count for text, count in counts.items() if text in test_texts),
        "removed_rows": len(train) - len(prepared),
        "train_prepared_rows": len(prepared),
    }
    return prepared, audit
