from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np

@dataclass
class ModelMetadata:
    name: str
    version: str
    feature_names: list[str]            = field(default_factory=list)
    target_name: str                    = ""
    metrics: dict[str, float]           = field(default_factory=dict)
    hyperparams: dict[str, Any]         = field(default_factory=dict)
    loaded_at: datetime                 = field(default_factory=datetime.utcnow)
    trained_at: datetime | None         = None
    extra: dict[str, Any]               = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "version":       self.version,
            "feature_names": self.feature_names,
            "target_name":   self.target_name,
            "metrics":       self.metrics,
            "hyperparams":   self.hyperparams,
            "loaded_at":     self.loaded_at.isoformat(),
            "trained_at":    self.trained_at.isoformat() if self.trained_at else None,
            "extra":         self.extra,
        }

    def validate_features(self, received: list[str]) -> None:
        if not self.feature_names:
            return  # sin schema definido, no valida

        expected = set(self.feature_names)
        got      = set(received)

        missing = expected - got
        extra   = got - expected

        errors = []
        if missing:
            errors.append(f"Features faltantes: {sorted(missing)}")
        if extra:
            errors.append(f"Features no esperados: {sorted(extra)}")

        if errors:
            raise ValueError(" | ".join(errors))

class ModelNotLoadedError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"El modelo '{name}' no fue cargado. Llamá load() primero.")


class PredictProbaNotSupportedError(NotImplementedError):
    def __init__(self, name: str) -> None:
        super().__init__(f"El modelo '{name}' no soporta predict_proba().")


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
        self._model: Any = None          # objeto sklearn / xgboost / torch / etc.

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las probabilidades por clase para cada fila de X.
        Shape esperado: (n_samples, n_classes)

        Solo sobreescribir si el modelo soporta probabilidades.
        Por defecto lanza PredictProbaNotSupportedError.
        """
        raise PredictProbaNotSupportedError(self._metadata.name)

    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        Predicción en chunks para datasets grandes.
        Por defecto simplemente llama a predict() de una sola vez.
        Sobreescribir si el modelo necesita manejo especial de memoria.

        Args:
            X:          array completo de features
            batch_size: tamaño de cada chunk

        Returns:
            array concatenado con todas las predicciones
        """
        if len(X) <= batch_size:
            return self.predict(X)

        results = []
        for start in range(0, len(X), batch_size):
            chunk = X[start : start + batch_size]
            results.append(self.predict(chunk))

        return np.concatenate(results)

    @property
    def is_loaded(self) -> bool:
        """True si el artefacto fue cargado exitosamente."""
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

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"name={self._metadata.name!r}, "
            f"version={self._metadata.version!r}, "
            f"status={status!r})"
        )
