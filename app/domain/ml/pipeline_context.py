from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BaseContext:
    model_name: str = ""
    version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    extra: dict[str, Any] = field(default_factory=dict)
    steps_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    tarea_programada_id: int | None = None
    ejecucion_id: int | None = None

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class TrainingContext(BaseContext):
    raw_data: pd.DataFrame | None = None
    clean_data: pd.DataFrame | None = None  # después de CleanStep


@dataclass
class PredictContext(BaseContext):
    model: Any = None  # el modelo cargado, para usar en pasos posteriores
    data_response: list[dict[str, Any]] | None = (
        None  # el resultado de la predicción, para usar en pasos posteriores
    )
