from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.domain.dtos.model_metadata_dto import ModelMetadataDTO

@dataclass
class ModelMetadata:
    name: str
    version: str
    feature_names: list[str]            = field(default_factory=list)
    target_name: str                    = ""
    hyperparams: dict[str, Any]         = field(default_factory=dict)
    loaded_at: datetime                 = field(default_factory=lambda: datetime.now(timezone.utc))
    trained_at: datetime | None         = None
    extra: dict[str, Any]               = field(default_factory=dict)
    path_model : str = ""

    def to_dict(self) -> dict:
         return {
            "name":          self.name,
            "version":       self.version,
            "feature_names": self.feature_names,
            "target_name":   self.target_name,
            "hyperparams":   self.hyperparams,
            "loaded_at":     self.loaded_at.isoformat(),
            "trained_at":    self.trained_at.isoformat() if self.trained_at else None,
            "extra":         self.extra,
            "path_model":    self.path_model,
        }
         
    def to_dto(self) -> "ModelMetadataDTO":
        return ModelMetadataDTO(
            name=self.name,
            version=self.version,
            feature_names=self.feature_names,
            target_name=self.target_name,
            hyperparams=self.hyperparams,
            loaded_at=self.loaded_at,
            trained_at=self.trained_at,
            path_model=self.path_model
        )