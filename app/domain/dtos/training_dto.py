from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainRequestDTO:
    model_name: str = "pedido_sugerido"
    version: str = "1.0"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    data_source_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResponseDTO:
    model_name: str
    version: str
    steps_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    success: bool = False
