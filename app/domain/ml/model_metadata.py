from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelMetadata:
    name: str
    version: str
    feature_names: list[str]        = field(default_factory=list)
    target_name: str                = ""
    hyperparams: dict[str, Any]     = field(default_factory=dict)
    loaded_at: datetime             = field(default_factory=lambda: datetime.now(timezone.utc))
    trained_at: datetime | None     = None
    extra: dict[str, Any]           = field(default_factory=dict)
    path_model: str                 = ""