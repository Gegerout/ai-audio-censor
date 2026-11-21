"""Проверки обучения, позиций слов и сохранения модели"""

import mlflow.pyfunc
import pandas as pd
import pytest
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import Array

from ml.data import Sample
from ml.model import SpanModel


@pytest.fixture
def samples():
    """Слова встречаются в разных текстах, метка известна заранее"""
    return [
        Sample(row_id=index, text=f"idiot nice {index}", positions=list(range(5)))
        for index in range(10)
    ]


@pytest.mark.parametrize("variant", ["dictionary", "word_logreg", "context_logreg"])
def test_model_preserves_offsets_and_learns_only_train(variant, samples):
    """Регистр и Unicode не сдвигают позиции, неизвестные слова не попадают в словарь"""
    model = SpanModel(variant)
    model.fit(samples)
    texts = ["я😊 IDIOT nice", "", "!!!", "nice"]
    result = model.predict(pd.DataFrame({"text": texts}))["positions"].tolist()
    assert result == [list(range(3, 8)), [], [], []]
    if variant == "dictionary":
        assert model.vocabulary == {"idiot"}
    else:
        assert "word=я" not in model.pipeline["vectorizer"].vocabulary_


def test_dictionary_filters_rare_and_ambiguous_words():
    """Редкое слово и слово с малой долей положительных меток не включаются"""
    model = SpanModel("dictionary")
    model.fit(
        [
            Sample(row_id=0, text="rare mixed", positions=list(range(10))),
            Sample(row_id=1, text="mixed", positions=[]),
            Sample(row_id=2, text="mixed", positions=[]),
        ]
    )
    assert model.vocabulary == set()
    assert model.predict_positions(["rare mixed"]) == [[]]


def test_model_rejects_invalid_training_and_unfitted_prediction():
    """Ошибки конфигурации и обучения не превращаются в пустые предсказания"""
    with pytest.raises(ValueError, match="Неизвестный"):
        SpanModel("other")
    model = SpanModel("dictionary")
    with pytest.raises(ValueError, match="не обучена"):
        model.predict_positions(["hello"])
    with pytest.raises(ValueError, match="нужны слова"):
        model.fit([Sample(row_id=0, text="hello", positions=[])])


@pytest.mark.parametrize("variant", ["dictionary", "word_logreg", "context_logreg"])
def test_model_round_trip(tmp_path, variant, samples):
    """Сохранённая модель принимает тексты и возвращает прежние позиции"""
    model = SpanModel(variant)
    model.fit(samples)
    frame = pd.DataFrame({"text": ["idiot nice", "", "😊 IDIOT"]})
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "text")]),
        outputs=Schema([ColSpec(Array("long"), "positions")]),
    )
    path = tmp_path / "model"
    mlflow.pyfunc.save_model(
        str(path), python_model=model, signature=signature, input_example=frame, pip_requirements=[]
    )
    loaded = mlflow.pyfunc.load_model(str(path))
    assert loaded.predict(frame)["positions"].tolist() == model.predict(frame)["positions"].tolist()
