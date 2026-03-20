from dataclasses import dataclass, field
from typing import Any

@dataclass
class PredictRequestDTO():
    model_name:  str  = "knn"
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictResponseDTO():
    model_name:     str
    predictions:    Any
    success:        bool = False