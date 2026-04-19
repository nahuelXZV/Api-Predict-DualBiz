from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelMetadata:
    name: str
    version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)
    path_model: str = ""
