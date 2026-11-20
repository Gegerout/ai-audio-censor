"""Восстановить выбранную словарную модель при первом запуске пустого реестра"""

import logging
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from ml.data import DATASET_REVISION, download_split, prepare_train, read_samples
from ml.evaluation import split_train
from ml.model import SpanModel
from src.config import get_settings
from src.logging_config import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    """Сохранить рабочий alias или восстановить его из данных второй работы"""
    settings = get_settings()
    setup_logging(settings.log_level)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    name, alias = settings.mlflow_model_name, settings.mlflow_model_alias
    try:
        client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        return
    versions = [v for v in client.search_model_versions() if v.name == name]
    if versions:
        for existing in versions:
            if existing.tags.get("bootstrap.alias") == alias:
                client.set_registered_model_alias(name, alias, existing.version)
                return
        raise ValueError(f"В реестре есть версии {name}, задайте рабочий alias {alias}")
    with tempfile.TemporaryDirectory() as temp:
        raw = Path("data/raw")
        if (
            not (raw / DATASET_REVISION / "train.csv").is_file()
            or not (raw / DATASET_REVISION / "test.csv").is_file()
        ):
            raw = Path(temp)
        train, test = (read_samples(download_split(split, raw)) for split in ("train", "test"))
        prepared, _ = prepare_train(train, test)
        samples, _ = split_train(prepared, 42)
        model = SpanModel("dictionary", 42)
        model.fit(samples)
        mlflow.set_experiment("toxic-spans")
        with mlflow.start_run(run_name="bootstrap_dictionary"):
            mlflow.log_params({"variant": "dictionary", "seed": 42, "revision": DATASET_REVISION})
            example = pd.DataFrame({"text": ["sample"]})
            saved = mlflow.pyfunc.log_model(
                name="model",
                python_model=model,
                code_paths=["ml"],
                input_example=example,
                signature=mlflow.models.infer_signature(example, model.predict(example)),
                pip_requirements=[
                    f"{pkg}=={version(pkg)}"
                    for pkg in ("mlflow", "pandas", "scikit-learn", "httpx")
                ],
            )
            registered = mlflow.register_model(
                saved.model_uri, name, tags={"bootstrap.alias": alias}
            )
            client.set_registered_model_alias(name, alias, registered.version)
            log.info(
                "model registered",
                extra={"model_alias": alias, "model_version": registered.version},
            )


if __name__ == "__main__":
    main()
