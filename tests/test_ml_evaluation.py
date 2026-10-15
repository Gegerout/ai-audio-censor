"""Проверки разделения выборок и расчёта метрик"""

import pytest

from ml.data import Sample
from ml.evaluation import evaluate, score_positions, split_train


def test_split_is_reproducible_disjoint_and_stratified():
    """Все тексты входят ровно в одну часть, доля непустой разметки сохраняется"""
    samples = [
        Sample(row_id=index, text=f"text {index}", positions=[0] if index % 2 else [])
        for index in range(50)
    ]
    train, validation = split_train(samples, 42)
    assert (train, validation) == split_train(samples, 42)
    assert (train, validation) != split_train(samples, 43)
    assert len(train) == 40 and len(validation) == 10
    assert {row["row_id"] for row in train}.isdisjoint(row["row_id"] for row in validation)
    assert sorted(row["row_id"] for row in train + validation) == list(range(50))
    assert sum(bool(row["positions"]) for row in validation) == 5


@pytest.mark.parametrize(
    ("expected", "predicted", "scores"),
    [
        ([], [], {"f1": 1.0, "precision": 1.0, "recall": 1.0}),
        ([], [0], {"f1": 0.0, "precision": 0.0, "recall": 0.0}),
        ([0], [], {"f1": 0.0, "precision": 0.0, "recall": 0.0}),
        ([0, 1], [1], {"f1": 2 / 3, "precision": 1.0, "recall": 0.5}),
        ([0, 1], [1, 1], {"f1": 2 / 3, "precision": 1.0, "recall": 0.5}),
    ],
)
def test_character_metrics_and_empty_annotations(expected, predicted, scores):
    """Правила частичного совпадения и пустых ответов одинаковы для всех моделей"""
    assert score_positions(expected, predicted) == scores


def test_evaluation_averages_by_text_and_counts_each_position():
    """Пустая разметка учитывается в общей оценке, но не в оценке размеченных текстов"""
    samples = [
        Sample(row_id=1, text="abc", positions=[0, 1]),
        Sample(row_id=2, text="xx", positions=[]),
    ]
    metrics, rows, matrix = evaluate(samples, [[1, 2], []])
    assert metrics == {
        "val_f1": 0.75,
        "val_precision": 0.75,
        "val_recall": 0.75,
        "val_f1_annotated": 0.5,
        "val_f1_micro": 0.5,
    }
    assert matrix == [[2, 1], [1, 1]]
    assert rows[0]["predicted_positions"] == [1, 2]
    assert rows[0]["row_id"] == 1


def test_evaluation_rejects_missing_or_invalid_predictions():
    """Неполные ответы и позиции вне текста не дают корректную метрику"""
    sample = Sample(row_id=0, text="a", positions=[])
    with pytest.raises(ValueError, match="Нужны"):
        evaluate([], [])
    with pytest.raises(ValueError, match="Нужны"):
        evaluate([sample], [])
    with pytest.raises(ValueError, match="границы"):
        evaluate([sample], [[1]])
    assert evaluate([sample], [[]])[0]["val_f1_annotated"] == 0
