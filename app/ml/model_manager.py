from __future__ import annotations
import pandas as pd
from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.dtos.model_metadata_dto import ModelMetadataDTO
from app.domain.dtos.training_dto import TrainResponseDTO
from app.domain.ml.model_registry import model_registry
from app.domain.ml.base_context import TrainingContext
from app.ml.training.pedido_sugerido.pipeline import build_pedido_sugerido_pipeline

def _get_train_pipelines() -> dict:
    return {
        "pedido_sugerido": build_pedido_sugerido_pipeline,
        # "svm": build_svm_pipeline,
    }

class ModelManager:

    def train(self, model_name:  str, version:str, hyperparams: dict = {}) -> TrainResponseDTO:
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

    def predict(self, model_name: str, data:dict) -> pd.DataFrame:
        logger.info("manager_predict_start", model=model_name)

        if not model_registry.exists(model_name):
            raise ModelNotFoundError(
                f"Modelo '{model_name}' no está en el registry. "
                "Ejecutá /train primero."
            )

        model = model_registry.get(model_name)
        return model.predict(data)

    def list_models(self) -> list[ModelMetadataDTO]:
        return model_registry.list_models()
    
model_manager = ModelManager()