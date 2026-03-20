from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.domain.dtos.training_dto import TrainResponseDTO
from app.domain.ml.base_model import BaseMLModel

@dataclass
class BaseContext:
    model_name:     str  = ""
    version:      str  = ""
    data_path:    str  = ""
    hyperparams:  dict[str, Any] = field(default_factory=dict)
    
    extra:        dict[str, Any]       = field(default_factory=dict)
    started_at:      datetime          = field(default_factory=datetime.utcnow)
    steps_executed:  list[str]         = field(default_factory=list)
    errors:          list[str]         = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def summary(self) -> TrainResponseDTO:
        return TrainResponseDTO(
            model_name=self.model_name,
            version=self.version,
            steps_executed=self.steps_executed,
            errors=self.errors,
            success=not self.has_errors
        )
    
@dataclass
class TrainingContext(BaseContext):
    raw_data:     pd.DataFrame | None  = None   
    clean_data:   pd.DataFrame | None  = None   # después de CleanStep
    X_train:      np.ndarray   | None  = None   # después de FeatureStep
    X_test:       np.ndarray   | None  = None
    y_train:      np.ndarray   | None  = None
    y_test:       np.ndarray   | None  = None
    trained_model: Any                 = None   # después de TrainStep

  
    
@dataclass
class PredictContext(BaseContext):
    data:        dict[str, Any]       = field(default_factory=dict)
    model:  Any = None   # el modelo cargado, para usar en pasos posteriores
    data_response : pd.DataFrame | None = None  # el resultado de la predicción, para usar en pasos posteriores