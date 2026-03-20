from dataclasses import dataclass, field
from typing import Any

@dataclass
class PredictResponseDTO():
    model_name:  str
    version:     str
    records:     int
    errors:      list[str] = field(default_factory=list)
    success:     bool = False
