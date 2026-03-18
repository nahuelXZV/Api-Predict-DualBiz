from __future__ import annotations

import threading
from typing import Iterator

from app.core.exceptions import ModelAlreadyExistsError, ModelNotFoundError, ModelNotReadyError
from app.domain.ml.base_model import BaseMLModel
from app.core.logging import logger

class ModelRegistry:
    """
    Registro central de modelos ML en memoria.

    Responsabilidades:
      - Almacenar instancias de BaseMLModel indexadas por nombre.
      - Garantizar acceso thread-safe (hot reload sin downtime).
      - Proveer consultas por nombre, listado y limpieza de memoria.

    Qué NO hace:
      - No carga archivos .pkl — eso es responsabilidad de cada modelo y del RegisterStep.
      - No conoce rutas de disco ni storage.
      - No conoce FastAPI ni schedulers.

    Uso típico:
        # Al iniciar la app (lifespan)
        model = IrisClassifier(metadata)
        model.load("models/iris_v1.pkl")
        model_registry.register("iris", model)

        # En PredictionService o BatchPredictJob
        model = model_registry.get("iris")
        predictions = model.predict(X)

        # Hot reload sin downtime (desde AdminEndpoint o TrainingJob)
        new_model = IrisClassifier(new_metadata)
        new_model.load("models/iris_v2.pkl")
        model_registry.register("iris", new_model, allow_override=True)
    """

    def __init__(self) -> None:
        self._models: dict[str, BaseMLModel] = {}
        self._lock   = threading.Lock()
        
    def register(self, name: str, model: BaseMLModel, * ,allow_override: bool = True) -> None:
        if not model.is_loaded:
            raise ValueError(f"No se puede registrar '{name}': ""el modelo no fue cargado (llamá model.load() antes).")

        with self._lock:
            if not allow_override and name in self._models:
                raise ModelAlreadyExistsError(name)

            previous_version = (
                self._models[name].version if name in self._models else None
            )
            self._models[name] = model

        logger.info(
            "model_registered",
            name        = name,
            version     = model.version,
            replaced    = previous_version,
            loaded_at   = model.metadata.loaded_at.isoformat(),
        )

    def unload(self, name: str) -> None:
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(name)
            del self._models[name]

        logger.info("model_unloaded", name=name)

    def clear(self) -> None:
        with self._lock:
            names = list(self._models.keys())
            self._models.clear()

        logger.info("registry_cleared", removed=names)

    def get(self, name: str) -> BaseMLModel:
        model = self._models.get(name)

        if model is None:
            raise ModelNotFoundError(name)

        if not model.is_loaded:
            raise ModelNotReadyError(name)

        return model

    def get_or_none(self, name: str) -> BaseMLModel | None:
        return self._models.get(name)

    def exists(self, name: str) -> bool:
        return name in self._models

    def list_models(self) -> list[dict]:
        with self._lock:
            snapshot = list(self._models.items())

        return [
            {
                "name":          name,
                "version":       model.version,
                "is_loaded":     model.is_loaded,
                "loaded_at":     model.metadata.loaded_at.isoformat(),
                "feature_names": model.metadata.feature_names,
                "metrics":       model.metadata.metrics,
            }
            for name, model in snapshot
        ]

    def __iter__(self) -> Iterator[tuple[str, BaseMLModel]]:
        with self._lock:
            items = list(self._models.items())
        return iter(items)

model_registry = ModelRegistry()
