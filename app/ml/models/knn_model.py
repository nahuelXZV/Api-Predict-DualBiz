
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.domain.core.exceptions import ModelNotLoadedError
from app.domain.ml.base_model import BaseMLModel
from app.domain.ml.model_metadata import ModelMetadata


class KnnModel(BaseMLModel):
    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__(metadata)
        self._model: RandomForestClassifier | None = None

    def load(self, path: str) -> None:
        loaded = joblib.load(path)

        # if not isinstance(loaded, RandomForestClassifier):
        #     raise ValueError(
        #         f"IrisClassifier esperaba un RandomForestClassifier, "
        #         f"pero el archivo contiene un {type(loaded).__name__}. "
        #         f"Verificá que el archivo '{path}' es el correcto."
        #     )
        self._model = loaded

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ModelNotLoadedError(self.metadata.name)

        return self._model.predict(X)