from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from app.core.exceptions import ModelNotLoadedError
from app.domain.ml.model_metadata import ModelMetadata
       
class BaseMLModel(ABC):
    """
    Contrato base que deben implementar todos los modelos del sistema.

    Responsabilidades:
      - Cargar el artefacto entrenado desde disco o storage (load).
      - Ejecutar predicción sobre un array de features (predict).
      - Exponer probabilidades si el modelo las soporta (predict_proba).
      - Guardar metadatos accesibles para el Registry y los endpoints.

    Qué NO hace:
      - No conoce FastAPI ni HTTP.
      - No sabe de base de datos ni schedulers.
      - No hace preprocesamiento — eso es responsabilidad del FeatureStep.

    Uso básico:
        model = IrisClassifier(metadata)
        model.load("models/iris_v1.pkl")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    """

    def __init__(self, metadata: ModelMetadata) -> None:
        self._metadata: ModelMetadata = metadata
        self._model: Any = None         

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Carga el artefacto entrenado desde `path` y lo guarda en self._model.
        Después de llamar load(), is_loaded debe devolver True.

        Args:
            path: ruta al archivo (local o montado). Ej: "models/iris_v1.pkl"
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las clases o valores predichos para cada fila de X.

        Args:
            X: array de shape (n_samples, n_features)

        Returns:
            array de shape (n_samples,)
        """
        ...

    @abstractmethod
    def training(self, x: np.ndarray) -> np.ndarray:
        ...

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def version(self) -> str:
        return self._metadata.version

    def _assert_loaded(self) -> None:
        """
        Lanza ModelNotLoadedError si el modelo no está cargado.
        Llamar al inicio de predict() y predict_proba() en las subclases.

        Ejemplo:
            def predict(self, X):
                self._assert_loaded()
                return self._model.predict(X)
        """
        if not self.is_loaded:
            raise ModelNotLoadedError(self._metadata.name)
