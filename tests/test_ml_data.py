"""Проверки целостности датасета и защиты от пересечений выборок"""

import csv
import hashlib

import httpx
import pytest

from ml.data import (
    DATASET_REVISION,
    FILE_HASHES,
    Sample,
    download_split,
    parse_positions,
    prepare_train,
    read_samples,
    source_url,
)


def test_download_checks_hash_and_reuses_cache(tmp_path, monkeypatch):
    """Повторная загрузка использует проверенный файл без обращения к сети"""
    content = b"text_of_post,position\nhello,[]\n"
    monkeypatch.setitem(FILE_HASHES, "train", hashlib.sha256(content).hexdigest())
    calls = []

    def get(url, **kwargs):
        """Вернуть файл вместо сетевого запроса"""
        calls.append(url)
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    path = download_split("train", tmp_path)
    assert path.read_bytes() == content
    assert download_split("train", tmp_path) == path
    assert calls == [source_url("train")]
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Контрольная сумма"):
        download_split("train", tmp_path)


def test_download_rejects_wrong_remote_content(tmp_path, monkeypatch):
    """Неверный ответ сервера не сохраняется как исходный датасет"""

    def get(url, **kwargs):
        """Вернуть повреждённое содержимое"""
        return httpx.Response(200, content=b"wrong", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    with pytest.raises(ValueError, match="Контрольная сумма"):
        download_split("train", tmp_path)
    assert not (tmp_path / DATASET_REVISION / "train.csv").exists()


def test_source_is_pinned():
    """Источник содержит версию репозитория, неизвестные выборки отклоняются"""
    assert f"/resolve/{DATASET_REVISION}/test.csv" in source_url("test")
    with pytest.raises(ValueError, match="Неизвестная выборка"):
        source_url("other")


@pytest.mark.parametrize("value", ["oops", "[", "{}", "[True]", "[1.5]", "[-1]", "[3]"])
def test_positions_reject_invalid_annotations(value):
    """Разметка принимает только целые позиции внутри текста"""
    with pytest.raises(ValueError):
        parse_positions(value, "abc")


def test_positions_preserve_unicode_offsets():
    """Позиции считаются по символам, а не по байтам"""
    assert parse_positions("[2, 1, 1]", "я😊z") == [1, 2]
    assert parse_positions("[]", "text") == []


def test_read_samples_preserves_text_and_row_id(tmp_path):
    """Чтение сохраняет пробелы и переносы, от которых зависят позиции"""
    path = tmp_path / "train.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text_of_post", "position"])
        writer.writerow([" A\nБ ", "[3]"])
        writer.writerow(["second", "[]"])
    assert read_samples(path) == [
        Sample(row_id=0, text=" A\nБ ", positions=[3]),
        Sample(row_id=1, text="second", positions=[]),
    ]


@pytest.mark.parametrize(
    "content",
    [
        "wrong,columns\n",
        "text_of_post,position\n",
        "text_of_post,position\n ,[]\n",
        "text_of_post,position\nhello\n",
    ],
)
def test_read_samples_rejects_missing_data(tmp_path, content):
    """Пустая или неполная таблица не проходит проверку"""
    path = tmp_path / "train.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_samples(path)


def test_prepare_train_removes_duplicates_and_test_overlap():
    """Нормализация устраняет утечку, но не изменяет тексты и разметку"""
    train = [
        Sample(row_id=0, text="Repeat text", positions=[]),
        Sample(row_id=1, text=" REPEAT  text ", positions=[1]),
        Sample(row_id=2, text="Held out", positions=[]),
        Sample(row_id=3, text=" Unique\nтекст ", positions=[8]),
    ]
    test = [Sample(row_id=0, text="HELD OUT", positions=[])]
    prepared, audit = prepare_train(train, test)
    assert prepared == [train[3]]
    assert len(train) == 4
    assert test[0]["text"] == "HELD OUT"
    assert audit["train_duplicate_groups"] == 1
    assert audit["train_duplicate_rows"] == 2
    assert audit["overlap_rows"] == 1
    assert audit["removed_rows"] == 3


def test_prepare_train_rejects_empty_result():
    """Полное пересечение выборок не выдаётся за готовый датасет"""
    samples = [Sample(row_id=0, text="same", positions=[])]
    with pytest.raises(ValueError, match="выборка пуста"):
        prepare_train(samples, samples)
