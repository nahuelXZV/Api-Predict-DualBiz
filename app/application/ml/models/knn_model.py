import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from app.domain.ml.base_model import BaseMLModel
from app.domain.ml.model_metadata import ModelMetadata


# class IrisClassifier(BaseMLModel):
#     def __init__(self, metadata: ModelMetadata) -> None:
#         super().__init__(metadata)
#         self._model: RandomForestClassifier | None = None

#     def load(self, path: str) -> None:
#         loaded = joblib.load(path)

#         if not isinstance(loaded, RandomForestClassifier):
#             raise ValueError(
#                 f"IrisClassifier esperaba un RandomForestClassifier, "
#                 f"pero el archivo contiene un {type(loaded).__name__}. "
#                 f"Verificá que el archivo '{path}' es el correcto."
#             )

#         self._model = loaded

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         self._assert_loaded()
#         return None