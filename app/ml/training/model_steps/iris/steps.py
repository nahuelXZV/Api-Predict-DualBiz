import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.domain.base_step import BaseTrainingStep
from app.ml.training.context import TrainingContext
from app.ml.registry import model_registry
from app.domain.base_model import ModelMetadata
from app.core.logging import logger

class LoadDataStep(BaseTrainingStep):
    def __init__(
        self,
        drop_na_subset: list[str] | None        = None,
        duplicate_subset: list[str] | None      = None,
    ) -> None:
        self._drop_na_subset     = drop_na_subset
        self._duplicate_subset   = duplicate_subset
        
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        
        return ctx
