import joblib
import pandas as pd

from app.domain.core.exceptions import ModelNotLoadedError
from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_model import BaseMLModel
from app.domain.ml.model_metadata import ModelMetadata
from app.ml.predict.pedido_sugerido.pipeline import predict_pedido_sugerido_pipeline


class PedidoSugeridoModel(BaseMLModel):
    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__(metadata)

    def load(self, path: str) -> None:
        loaded = joblib.load(path)
        self._model = loaded

    def predict(self, data: dict) -> pd.DataFrame:
        if self._model is None:
            raise ModelNotLoadedError(self.metadata.name)
        
        ctx = PredictContext(
            model_name=self.metadata.name,
            version=self.metadata.version,
            hyperparams=self.metadata.hyperparams,
            extra=self.metadata.extra,
            model=self._model,
            data=data,
        )

        pipeline = predict_pedido_sugerido_pipeline()
        ctx = pipeline.run(ctx)
        response = ctx.data_response
        return response
