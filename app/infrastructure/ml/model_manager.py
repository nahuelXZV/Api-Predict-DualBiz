from __future__ import annotations
import app.infrastructure.ml.training  # noqa: F401 — activa el auto-registro de pipelines
from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.dtos.training_dto import TrainRequestDTO, TrainResponseDTO
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.domain.ml.pipeline_context import TrainingContext
from app.infrastructure.ml.data_sources.data_source_factory import DataSourceFactory
from app.infrastructure.ml.pipeline_registry import get_training_pipeline


class ModelManager:
    def train(self, request: TrainRequestDTO) -> TrainResponseDTO:
        logger.info(
            "manager_train_start", model=request.model_name, version=request.version
        )
        try:
            PipelineClass = get_training_pipeline(request.model_name)
            data_source = DataSourceFactory.build(request.data_source_config)

            ctx = TrainingContext(
                model_name=request.model_name,
                version=request.version,
                hyperparams=request.hyperparams,
            )

            pipeline = PipelineClass()
            pipeline.set_datasource(data_source)
            result = pipeline.run(ctx)

            return TrainResponseDTO(
                model_name=result.model_name,
                version=result.version,
                steps_executed=result.steps_executed,
                errors=result.errors,
                success=not result.has_errors,
            )
        except Exception as e:
            logger.error(
                "manager_train_error",
                model=request.model_name,
                version=request.version,
                error=str(e),
            )
            return TrainResponseDTO(
                success=False,
                errors=[str(e)],
                model_name=request.model_name,
                version=request.version,
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
            return {
                "error": str(e),
                "message": f"Error al predecir con el modelo '{model_name}'. Ver logs para más detalles.",
            }

    def list_models(self) -> list[ModelMetadata]:
        return model_registry.list_models()


model_manager = ModelManager()
