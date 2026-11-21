"""Разделение данных и оценка совпадения предсказанных и размеченных позиций"""

from sklearn.model_selection import train_test_split

from ml.data import Sample


def split_train(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    """Отделить пятую часть текстов с сохранением доли непустой разметки"""
    train, validation = train_test_split(
        samples,
        test_size=0.2,
        random_state=seed,
        stratify=[bool(sample["positions"]) for sample in samples],
    )
    return train, validation


def score_positions(expected: list[int], predicted: list[int]) -> dict[str, float]:
    """Оценить один текст, считая совпадение двух пустых разметок правильным"""
    truth, found = set(expected), set(predicted)
    if not truth and not found:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    correct = len(truth & found)
    return {
        "f1": 2 * correct / (len(truth) + len(found)),
        "precision": correct / len(found) if found else 0.0,
        "recall": correct / len(truth) if truth else 0.0,
    }


def evaluate(
    samples: list[Sample], predictions: list[list[int]]
) -> tuple[dict[str, float], list[dict], list[list[int]]]:
    """Усреднить оценки по текстам и собрать ошибки на уровне позиций"""
    if not samples or len(samples) != len(predictions):
        raise ValueError("Нужны непустая выборка и одно предсказание для каждого текста")
    rows = []
    tn = fp = fn = tp = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        truth, found = set(sample["positions"]), set(prediction)
        if any(pos < 0 or pos >= len(sample["text"]) for pos in found):
            raise ValueError("Предсказанные позиции выходят за границы текста")
        scores = score_positions(sample["positions"], prediction)
        rows.append({**sample, "predicted_positions": prediction, **scores})
        tp += len(truth & found)
        fp += len(found - truth)
        fn += len(truth - found)
        tn += len(sample["text"]) - len(truth | found)
    metrics = {f"val_{key}": sum(row[key] for row in rows) / len(rows) for key in scores}
    annotated = [row for row in rows if row["positions"]]
    metrics["val_f1_annotated"] = sum(row["f1"] for row in annotated) / max(len(annotated), 1)
    metrics["val_f1_micro"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return metrics, rows, [[tn, fp], [fn, tp]]
