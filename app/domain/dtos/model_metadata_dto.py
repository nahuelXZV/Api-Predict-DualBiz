from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

@dataclass
class ModelMetadataDTO:
    name: str
    version: str
    feature_names: list[str]
    target_name: str
    loaded_at: datetime
    trained_at: datetime | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    path_model : str = ""