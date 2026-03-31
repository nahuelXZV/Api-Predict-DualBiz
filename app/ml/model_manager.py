from __future__ import annotations
from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.dtos.training_dto import TrainResponseDTO
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.domain.ml.base_context import TrainingContext
from app.ml.training.pedido_sugerido.pipeline import build_pedido_sugerido_pipeline


def _get_train_pipelines() -> dict:
    return {
        "pedido_sugerido": build_pedido_sugerido_pipeline,
        # "svm": build_svm_pipeline,
    }


class ModelManager:
    def train(
        self, model_name: str, version: str, hyperparams: dict = {}
    ) -> TrainResponseDTO:
        logger.info("manager_train_start", model=model_name, version=version)
        try:
            builder = _get_train_pipelines().get(model_name)
            if not builder:
                raise ValueError(
                    f"No hay training pipeline registrado para '{model_name}'. "
                    f"Modelos disponibles: {list(_get_train_pipelines().keys())}"
                )

            ctx = TrainingContext(
                model_name=model_name,
                version=version,
                hyperparams=hyperparams,
            )

            pipeline = builder()
            result = pipeline.run(ctx)

            return result.summary()
        except Exception as e:
            logger.error(
                "manager_train_error", model=model_name, version=version, error=str(e)
            )
            return TrainResponseDTO(
                success=False, errors=[str(e)], model_name=model_name, version=version
            )

    def predict(self, model_name: str, data: dict) -> dict:
        logger.info("manager_predict_start", model=model_name)

        try:
            if not model_registry.exists(model_name):
                raise ModelNotFoundError(
                    f"Modelo '{model_name}' no está en el registry. Ejecutá /train primero."
                )

            model = model_registry.get(model_name)
            return model.predict(data)
        except Exception as e:
            logger.error("manager_predict_error", model=model_name, error=str(e))
            response_error = {
                "error": str(e),
                "message": f"Error al predecir con el modelo '{model_name}'. Ver logs para más detalles.",
            }
            return response_error

    def list_models(self) -> list[ModelMetadata]:
        return model_registry.list_models()


model_manager = ModelManager()
