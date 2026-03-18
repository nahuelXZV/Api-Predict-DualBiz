from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class ModelMetadata:
    name: str
    version: str
    feature_names: list[str]            = field(default_factory=list)
    target_name: str                    = ""
    metrics: dict[str, float]           = field(default_factory=dict)
    hyperparams: dict[str, Any]         = field(default_factory=dict)
    loaded_at: datetime                 = field(default_factory=datetime.utcnow)
    trained_at: datetime | None         = None
    extra: dict[str, Any]               = field(default_factory=dict)

    def to_dict(self) -> dict:
         return {
            "name":          self.name,
            "version":       self.version,
            "feature_names": self.feature_names,
            "target_name":   self.target_name,
            "metrics":       self.metrics,
            "hyperparams":   self.hyperparams,
            "loaded_at":     self.loaded_at.isoformat(),
            "trained_at":    self.trained_at.isoformat() if self.trained_at else None,
            "extra":         self.extra,
        }
