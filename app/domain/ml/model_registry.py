from __future__ import annotations

import threading
from typing import Iterator

from app.domain.core.exceptions import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from app.domain.ml.abstractions.ml_model_abc import MLModelABC
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.core.logging import logger


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, MLModelABC] = {}
        self._lock = threading.Lock()

    def register(
        self, name: str, model: MLModelABC, *, allow_override: bool = True
    ) -> None:
        if not model.is_loaded:
            raise ValueError(
                f"No se puede registrar '{name}': el modelo no fue cargado..."
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
            name=name,
            version=model.version,
            replaced=previous_version,
            loaded_at=model.metadata.loaded_at.isoformat(),
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

    def get(self, name: str) -> MLModelABC:
        model = self._models.get(name)

        if model is None:
            raise ModelNotFoundError(name)

        if not model.is_loaded:
            raise ModelNotReadyError(name)

        return model

    def get_or_none(self, name: str) -> MLModelABC | None:
        return self._models.get(name)

    def exists(self, name: str) -> bool:
        return name in self._models

    def list_models(self) -> list[ModelMetadata]:
        with self._lock:
            snapshot = list(self._models.items())

        return [model.metadata for name, model in snapshot]

    def __iter__(self) -> Iterator[tuple[str, MLModelABC]]:
        with self._lock:
            items = list(self._models.items())
        return iter(items)


model_registry = ModelRegistry()
