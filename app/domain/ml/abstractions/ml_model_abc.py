from abc import ABC, abstractmethod
from typing import Any

from app.domain.core.exceptions import ModelNotLoadedError
from app.domain.ml.model_metadata import ModelMetadata


class MLModelABC(ABC):
    def __init__(self, metadata: ModelMetadata) -> None:
        self._metadata: ModelMetadata = metadata
        self._model: Any = None

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def predict(self, data: dict) -> dict: ...

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
        if not self.is_loaded:
            raise ModelNotLoadedError(self._metadata.name)
