from __future__ import annotations
from app.domain.core.config import settings
from app.domain.core.exceptions import ModelNotFoundError
from app.domain.core.logging import logger
from app.domain.dtos.training_dto import TrainRequestDTO, TrainResponseDTO
from app.domain.ml.data_source import DataSource
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.domain.ml.base_context import TrainingContext
from app.infrastructure.ml.training.pedido_sugerido.pipeline import (
    build_pedido_sugerido_pipeline,
)
from app.infrastructure.ml.training.pedido_sugerido.queries import CONSULTA_BASE


def _get_train_pipelines() -> dict:
    return {
        "pedido_sugerido": build_pedido_sugerido_pipeline,
        # "svm": build_svm_pipeline,
    }


_DEFAULT_QUERIES: dict[str, str] = {
    "pedido_sugerido": CONSULTA_BASE,
}


def _build_data_source(config: dict, model_name: str) -> DataSource:
    source_type = config.get("type")
    params: dict = config.get("params") or {}

    if source_type == "sqlserver":
        from app.infrastructure.ml.data_sources.sqlserver_data_source import (
            SqlServerDataSource,
        )

        conn_str = params.get("connection_string") or settings.ml_db_connection_string
        query = params.get("query") or _DEFAULT_QUERIES.get(model_name, "")
        if not query:
            raise ValueError(f"No hay query por defecto para el modelo '{model_name}'.")
        return SqlServerDataSource(conn_str, query)

    if source_type == "csv":
        from app.infrastructure.ml.data_sources.csv_data_source import CsvDataSource

        path = params.get("path")
        if not path:
            raise ValueError("El datasource 'csv' requiere el parámetro 'path'.")
        return CsvDataSource(
            path,
            separator=params.get("separator", ","),
            encoding=params.get("encoding", "utf-8"),
        )

    raise ValueError(
        f"Tipo de datasource desconocido: '{source_type}'. Opciones: sqlserver, csv."
    )


class ModelManager:
    def train(self, request: TrainRequestDTO) -> TrainResponseDTO:
        logger.info(
            "manager_train_start", model=request.model_name, version=request.version
        )
        try:
            builder = _get_train_pipelines().get(request.model_name)
            if not builder:
                raise ValueError(
                    f"No hay training pipeline registrado para '{request.model_name}'. "
                    f"Modelos disponibles: {list(_get_train_pipelines().keys())}"
                )

            data_source = _build_data_source(
                request.data_source_config, request.model_name
            )

            ctx = TrainingContext(
                model_name=request.model_name,
                version=request.version,
                hyperparams=request.hyperparams,
            )

            pipeline = builder(data_source)
            result = pipeline.run(ctx)

            return result.summary()
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
            response_error = {
                "error": str(e),
                "message": f"Error al predecir con el modelo '{model_name}'. Ver logs para más detalles.",
            }
            return response_error

    def list_models(self) -> list[ModelMetadata]:
        return model_registry.list_models()


model_manager = ModelManager()
