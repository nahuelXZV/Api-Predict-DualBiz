import joblib

from app.domain.core.exceptions import ModelNotLoadedError
from app.domain.ml.pipeline_context import PredictContext
from app.domain.ml.abstractions.ml_model_abc import MLModelABC
from app.domain.ml.model_metadata import ModelMetadata
from app.infrastructure.ml.predict.pedido_sugerido.pipeline import (
    PedidoSugeridoPredictPipeline,
)


class PedidoSugeridoModel(MLModelABC):
    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__(metadata)

    def load(self, path: str) -> None:
        loaded = joblib.load(path)
        self._model = loaded

    def predict(self, data: dict) -> dict:
        if self._model is None:
            raise ModelNotLoadedError(self.metadata.name)

        ctx = PredictContext(
            model_name=self.metadata.name,
            version=self.metadata.version,
            hyperparams=self.metadata.hyperparams,
            extra=self.metadata.extra,
            model=self._model,
            parameters=data,
        )

        pipeline = PedidoSugeridoPredictPipeline()
        ctx = pipeline.run(ctx)
        response = ctx.data_response
        return response
