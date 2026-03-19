from __future__ import annotations
import numpy as np
import pandas as pd
from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.ml.model_registry import model_registry
from app.domain.ml.base_context import TrainingContext
from app.ml.training.knn.pipeline import build_knn_pipeline

def _get_train_pipelines() -> dict:
    return {
        "knn": build_knn_pipeline,
        # "svm": build_svm_pipeline,
    }

class ModelManager:

    def train(self, model_name:  str, version:str, hyperparams: dict = {}) -> dict:
        logger.info("manager_train_start", model=model_name, version=version)

        builder = _get_train_pipelines().get(model_name)
        if not builder:
            raise ValueError(
                f"No hay training pipeline registrado para '{model_name}'. "
                f"Modelos disponibles: {list(_get_train_pipelines().keys())}"
            )

        ctx = TrainingContext(
            model_name  = model_name,
            version     = version,
            hyperparams = hyperparams,
        )

        pipeline = builder()
        result   = pipeline.run(ctx)

        return result.summary()

    def predict(self, model_name: str, version: str, data: pd.DataFrame) -> np.ndarray:
        logger.info("manager_predict_start", model=model_name, version=version)

        if not model_registry.exists(model_name):
            raise ModelNotFoundError(
                f"Modelo '{model_name}' no está en el registry. "
                "Ejecutá /train primero."
            )

        model = model_registry.get(model_name)
        return model.predict(data)

model_manager = ModelManager()