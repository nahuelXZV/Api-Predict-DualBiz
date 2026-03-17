from __future__ import annotations

import threading
from datetime import datetime
from typing import Iterator

from app.ml.base_model import BaseMLModel
from app.core.logging import logger

# Excepciones del registry
class ModelNotFoundError(KeyError):
    """El modelo solicitado no existe en el registry."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Modelo '{name}' no encontrado en el registry.")


class ModelNotReadyError(RuntimeError):
    """El modelo existe pero no fue cargado correctamente."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Modelo '{name}' existe pero no está listo (is_loaded=False).")


class ModelAlreadyExistsError(ValueError):
    """Se intenta registrar un nombre que ya existe sin allow_override."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"Modelo '{name}' ya está registrado. "
            "Usá allow_override=True para reemplazarlo."
        )


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
        
    def register(
        self,
        name: str,
        model: BaseMLModel,
        *,
        allow_override: bool = True,
    ) -> None:
        """
        Registra un modelo bajo `name`.

        Args:
            name:           clave de acceso. Ej: "iris", "fraud_v2"
            model:          instancia de BaseMLModel ya cargada (is_loaded=True)
            allow_override: si es False y el nombre ya existe, lanza
                            ModelAlreadyExistsError. Por defecto True para
                            permitir hot reload sin downtime.

        Raises:
            ModelAlreadyExistsError: si allow_override=False y ya existe.
            ValueError:              si el modelo no está cargado.
        """
        if not model.is_loaded:
            raise ValueError(
                f"No se puede registrar '{name}': "
                "el modelo no fue cargado (llamá model.load() antes)."
            )

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
        """
        Elimina un modelo del registry y libera su memoria.
        Útil para remover versiones viejas después de un hot reload.

        Args:
            name: nombre del modelo a eliminar.

        Raises:
            ModelNotFoundError: si el nombre no existe.
        """
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(name)
            del self._models[name]

        logger.info("model_unloaded", name=name)

    def clear(self) -> None:
        """
        Elimina todos los modelos del registry.
        Útil en tests para garantizar aislamiento entre casos.
        """
        with self._lock:
            names = list(self._models.keys())
            self._models.clear()

        logger.info("registry_cleared", removed=names)

    def get(self, name: str) -> BaseMLModel:
        """
        Retorna el modelo registrado bajo `name`.

        Args:
            name: clave usada al registrar. Ej: "iris"

        Returns:
            Instancia de BaseMLModel lista para predecir.

        Raises:
            ModelNotFoundError: si el nombre no existe.
            ModelNotReadyError: si existe pero is_loaded=False (estado inconsistente).
        """
        model = self._models.get(name)

        if model is None:
            raise ModelNotFoundError(name)

        if not model.is_loaded:
            raise ModelNotReadyError(name)

        return model

    def get_or_none(self, name: str) -> BaseMLModel | None:
        """
        Igual que get() pero retorna None en lugar de lanzar excepción.
        Útil cuando querés verificar existencia sin try/except.
        """
        return self._models.get(name)

    def exists(self, name: str) -> bool:
        """True si el nombre está registrado (independientemente de is_loaded)."""
        return name in self._models

    # Consultas e inspección
    def list_models(self) -> list[dict]:
        """
        Retorna información resumida de todos los modelos registrados.
        Usado por GET /api/v1/models y JobExecution logs.

        Returns:
            Lista de dicts con name, version, is_loaded, loaded_at, metrics.
        """
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

    def names(self) -> list[str]:
        """Lista de nombres registrados."""
        with self._lock:
            return list(self._models.keys())

    def count(self) -> int:
        """Cantidad de modelos en el registry."""
        return len(self._models)

    def snapshot_at(self) -> datetime:
        """Timestamp actual — útil para health checks."""
        return datetime.utcnow()

    # Iteración
    def __iter__(self) -> Iterator[tuple[str, BaseMLModel]]:
        """Permite iterar sobre (name, model) como un dict."""
        with self._lock:
            items = list(self._models.items())
        return iter(items)

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    # Representación
    def __repr__(self) -> str:
        names = self.names()
        return f"ModelRegistry(models={names})"


# ---------------------------------------------------------------------------
# Instancia global — importar desde aquí en toda la app
# ---------------------------------------------------------------------------
#
# USO:
#   from app.ml.registry import model_registry
#
# TESTS:
#   Para aislar tests, llamar model_registry.clear() en el fixture de teardown,
#   o crear una instancia local: registry = ModelRegistry()
#
model_registry = ModelRegistry()
