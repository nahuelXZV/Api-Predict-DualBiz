from __future__ import annotations

import app.application.ml.pipelines.training  # noqa: F401 — activa el auto-registro de pipelines

from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.dtos.training_dto import TrainRequestDTO, TrainResponseDTO
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import ModelRegistry, model_registry
from app.domain.ml.pipeline_context import TrainingContext
from app.application.ml.pipeline_registry import get_training_pipeline
from app.infrastructure.data_sources.data_source_factory import DataSourceFactory
from app.application.services.version_modelo_service import (
    VersionModeloService,
    version_modelo_service,
)


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        factory: type[DataSourceFactory],
        version_service: VersionModeloService,
    ) -> None:
        self._registry = registry
        self._factory = factory
        self._version_service = version_service

    def train(self, request: TrainRequestDTO) -> TrainResponseDTO:
        logger.info(
            "manager_train_start", model=request.model_name, version=request.version
        )
        try:
            logger.info("manager_train_get_pipeline", model=request.model_name)
            PipelineClass = get_training_pipeline(request.model_name)

            logger.info("manager_train_build_datasource", model=request.model_name)
            data_source = self._factory.build(request.parameters)

            ctx = TrainingContext(
                model_name=request.model_name,
                version=request.version,
                parameters=request.parameters,
                tarea_programada_id=request.tarea_programada_id,
                ejecucion_id=request.ejecucion_id,
            )
            pipeline = PipelineClass()

            logger.info("manager_train_pipeline_set_datasource", model=request.model_name)
            pipeline.set_datasource(data_source)
            
            logger.info("manager_train_pipeline_run", model=request.model_name)
            result = pipeline.run(ctx)

            if not result.has_errors:
                path_model = result.extra.get("path_model")
                if path_model:
                    self._version_service.save_new_version(
                        result, str(path_model), request.parameters
                    )

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
            if not self._registry.exists(model_name):
                raise ModelNotFoundError(
                    f"Modelo '{model_name}' no está en el registry. Ejecutá /train primero."
                )
            model = self._registry.get(model_name)
            return model.predict(data)
        except Exception as e:
            logger.error("manager_predict_error", model=model_name, error=str(e))
            return {
                "error": str(e),
                "message": f"Error al predecir con el modelo '{model_name}'. Ver logs para más detalles.",
            }

    def list_models(self) -> list[ModelMetadata]:
        return self._registry.list_models()


model_manager = ModelManager(
    registry=model_registry,
    factory=DataSourceFactory,
    version_service=version_modelo_service,
)
