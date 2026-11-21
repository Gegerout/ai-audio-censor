"""Словарная модель и классификация слов с восстановлением позиций в тексте"""

import re
from collections import Counter

import pandas as pd
from mlflow.pyfunc import PythonModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.data import Sample


def word_features(words: list[str], index: int, context: bool) -> dict[str, str]:
    """Описать слово и при необходимости добавить соседей и части слова"""
    word = words[index]
    features = {"word": word}
    if context:
        features["previous"] = words[index - 1] if index else "<START>"
        features["next"] = words[index + 1] if index + 1 < len(words) else "<END>"
        for size in (2, 3, 4):
            features[f"prefix_{size}"] = word[:size]
            features[f"suffix_{size}"] = word[-size:]
    return features


class SpanModel(PythonModel):
    """Предсказывать позиции токсичных фрагментов через метки отдельных слов"""

    def __init__(self, variant: str, seed: int = 42) -> None:
        if variant not in {"dictionary", "word_logreg", "context_logreg"}:
            raise ValueError(f"Неизвестный вариант модели {variant}")
        self.variant = variant
        self.seed = seed
        self.vocabulary: set[str] | None = None
        self.pipeline: Pipeline | None = None

    def fit(self, samples: list[Sample]) -> None:
        """Построить словарь или обучить классификатор только на переданных данных"""
        totals: Counter[str] = Counter()
        marked: Counter[str] = Counter()
        features = []
        labels = []
        for sample in samples:
            tokens = list(re.finditer(r"\b\w+\b", sample["text"]))
            words = [token.group().casefold() for token in tokens]
            positions = set(sample["positions"])
            for index, token in enumerate(tokens):
                label = int(any(pos in positions for pos in range(token.start(), token.end())))
                totals[words[index]] += 1
                marked[words[index]] += label
                labels.append(label)
                if self.variant != "dictionary":
                    features.append(word_features(words, index, self.variant == "context_logreg"))
        if set(labels) != {0, 1}:
            raise ValueError("Для обучения нужны слова с разметкой и без неё")
        if self.variant == "dictionary":
            self.vocabulary = {
                word for word, count in totals.items() if count >= 2 and marked[word] / count >= 0.5
            }
        else:
            self.pipeline = Pipeline(
                [
                    ("vectorizer", DictVectorizer()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=500,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            self.pipeline.fit(features, labels)

    def predict_positions(self, texts: list[str]) -> list[list[int]]:
        """Вернуть позиции в исходных строках без изменения регистра и пробелов"""
        if self.vocabulary is None and self.pipeline is None:
            raise ValueError("Модель ещё не обучена")
        predictions = []
        for text in texts:
            tokens = list(re.finditer(r"\b\w+\b", text))
            words = [token.group().casefold() for token in tokens]
            if self.vocabulary is not None:
                labels = [word in self.vocabulary for word in words]
            elif tokens:
                features = [
                    word_features(words, index, self.variant == "context_logreg")
                    for index in range(len(words))
                ]
                labels = self.pipeline.predict_proba(features)[:, 1] >= 0.5
            else:
                labels = []
            predictions.append(
                [
                    pos
                    for token, label in zip(tokens, labels, strict=True)
                    if label
                    for pos in range(token.start(), token.end())
                ]
            )
        return predictions

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """Принять таблицу текстов через общий интерфейс MLflow"""
        return pd.DataFrame({"positions": self.predict_positions(model_input["text"].tolist())})
