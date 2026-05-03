from dataclasses import dataclass, field
from typing import Any

from app.domain.models.tarea_programada import TareaProgramada


@dataclass
class TrainRequestDTO:
    tarea_programada: TareaProgramada | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    ejecucion_id: int | None = None


@dataclass
class TrainResponseDTO:
    model_name: str
    version: str
    steps_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    success: bool = False
