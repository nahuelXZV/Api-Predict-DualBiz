from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from app.domain.core.config import tz_now


@dataclass
class BaseContext:
    model_name: str = ""
    version: str = ""
    data_path: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    extra: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=tz_now)
    steps_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    tarea_programada_id: int | None = None

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class TrainingContext(BaseContext):
    raw_data: pd.DataFrame | None = None
    clean_data: pd.DataFrame | None = None  # después de CleanStep


@dataclass
class PredictContext(BaseContext):
    parameters: dict[str, Any] = field(default_factory=dict)
    model: Any = None  # el modelo cargado, para usar en pasos posteriores
    data_response: dict[str, Any] | None = (
        None  # el resultado de la predicción, para usar en pasos posteriores
    )
