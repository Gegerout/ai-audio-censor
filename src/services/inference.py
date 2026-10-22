"""Загрузка модели из реестра и обработка текстов"""

from dataclasses import dataclass
from numbers import Integral

import mlflow
import pandas as pd
from mlflow.pyfunc import PyFuncModel

from src.config import Settings
from src.schemas import ModelInfo, ProcessResponse, TextSpan


def positions_to_spans(text: str, positions: list[int]) -> list[TextSpan]:
    """Объединить соседние позиции в фрагменты исходного текста"""
    if any(
        not isinstance(pos, Integral) or isinstance(pos, bool) or not 0 <= pos < len(text)
        for pos in positions
    ):
        raise ValueError("Модель вернула недопустимые позиции в тексте")
    spans = []
    for pos in sorted(set(positions)):
        if spans and pos == spans[-1].end:
            spans[-1].end = int(pos) + 1
            spans[-1].text = text[spans[-1].start : spans[-1].end]
        else:
            spans.append(TextSpan(start=int(pos), end=int(pos) + 1, text=text[pos : pos + 1]))
    return spans


@dataclass(frozen=True)
class InferenceService:
    """Модель и сведения о версии, сохранённые на весь срок работы процесса"""

    model: PyFuncModel
    info: ModelInfo

    def process(self, text: str) -> ProcessResponse:
        """Выполнить inference уже загруженной моделью без обращения к MLflow"""
        prediction = self.model.predict(pd.DataFrame({"text": [text]}))
        if len(prediction) != 1:
            raise ValueError("Модель должна вернуть один результат для одного текста")
        positions = prediction.iloc[0]["positions"]
        return ProcessResponse(model=self.info, spans=positions_to_spans(text, positions))


def load_inference(settings: Settings) -> InferenceService:
    """Разрешить alias один раз и загрузить соответствующую версию модели"""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    selected = client.get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )
    if selected.status != "READY":
        raise ValueError("Версия модели ещё не готова к использованию")
    model = mlflow.pyfunc.load_model(f"models:/{settings.mlflow_model_name}/{selected.version}")
    return InferenceService(
        model=model,
        info=ModelInfo(
            name=settings.mlflow_model_name,
            version=str(selected.version),
            alias=settings.mlflow_model_alias,
            run_id=selected.run_id,
        ),
    )
