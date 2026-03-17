"""
TrainingContext — objeto compartido que viaja por todo el pipeline.

Empieza con la configuración inicial y se va enriqueciendo en cada step:
  - IngestStep      agrega raw_data
  - CleanStep       agrega clean_data
  - FeatureStep     agrega X_train, X_test, y_train, y_test
  - TrainStep       agrega trained_model
  - EvaluateStep    agrega metrics (y errors si no pasa el umbral)
  - RegisterStep    lee todo lo anterior para guardar y registrar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingContext:

    # ---- configuración inicial (se pasa al crear el contexto) ----
    model_name:   str  = ""
    version:      str  = ""
    data_path:    str  = ""
    hyperparams:  dict[str, Any] = field(default_factory=dict)

    # ---- se llena en cada step ----
    raw_data:     pd.DataFrame | None  = None   # después de IngestStep
    clean_data:   pd.DataFrame | None  = None   # después de CleanStep
    X_train:      np.ndarray   | None  = None   # después de FeatureStep
    X_test:       np.ndarray   | None  = None
    y_train:      np.ndarray   | None  = None
    y_test:       np.ndarray   | None  = None
    trained_model: Any                 = None   # después de TrainStep
    metrics:      dict[str, Any]       = field(default_factory=dict)

    # ---- bolsillo de extras por modelo (scaler, encoder, etc.) ----
    extra:        dict[str, Any]       = field(default_factory=dict)

    # ---- trazabilidad ----
    started_at:      datetime          = field(default_factory=datetime.utcnow)
    steps_executed:  list[str]         = field(default_factory=list)
    errors:          list[str]         = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def summary(self) -> dict:
        """Resumen compacto para logs y respuestas de API."""
        return {
            "model_name":     self.model_name,
            "version":        self.version,
            "steps_executed": self.steps_executed,
            "metrics":        self.metrics,
            "errors":         self.errors,
            "success":        not self.has_errors,
            "started_at":     self.started_at.isoformat(),
        }