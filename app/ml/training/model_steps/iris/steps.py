"""
Steps específicos de Iris.
Cada uno hereda del step base compartido y sobreescribe
solo lo que Iris necesita diferente.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.ml.training.base_step import BaseTrainingStep
from app.ml.training.steps.ingest import IngestStep
from app.ml.training.steps.clean import CleanStep
from app.ml.training.steps.feature_engineering import FeatureEngStep
from app.ml.training.steps.train import TrainStep
from app.ml.training.steps.evaluate import EvaluateStep
from app.ml.training.context import TrainingContext
from app.ml.registry import model_registry
from app.ml.base_model import ModelMetadata
from app.core.logging import logger


# ---------------------------------------------------------------------------
# 1. Ingest — igual al base, Iris lee CSV estándar
#    No necesita sobreescribir, usa IngestStep directamente.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Clean — agrega validaciones propias de Iris
# ---------------------------------------------------------------------------

class IrisCleanStep(CleanStep):
    """
    Hereda la limpieza base (NaN, duplicados) y agrega:
    - filtra species inválidas
    - filtra valores fuera de rango biológico
    """

    VALID_SPECIES = {"setosa", "versicolor", "virginica"}
    FEATURE_COLS  = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        # 1. Limpieza base: NaN y duplicados
        ctx = super().execute(ctx)
        df  = ctx.clean_data.copy()

        rows_before = len(df)

        # 2. Validar species
        if "species" in df.columns:
            df = df[df["species"].isin(self.VALID_SPECIES)]

        # 3. Validar rangos (valores negativos o cero no tienen sentido)
        for col in self.FEATURE_COLS:
            if col in df.columns:
                df = df[df[col] > 0]

        rows_after = len(df)
        logger.info(
            "iris_clean_done",
            rows_removed = rows_before - rows_after,
            rows_kept    = rows_after,
        )

        ctx.clean_data = df.reset_index(drop=True)
        return ctx


# ---------------------------------------------------------------------------
# 3. Feature Engineering — encode target + split + scale
# ---------------------------------------------------------------------------

class IrisFeatureStep(FeatureEngStep):
    """
    Hereda el split y scaler base.
    Agrega: encoding del target string → int (setosa=0, etc.)
    """

    def __init__(
        self,
        target_column: str = "species",
        test_size: float   = 0.2,
        random_state: int  = 42,
    ) -> None:
        super().__init__(target_column=target_column, test_size=test_size)
        self._random_state = random_state

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        df = ctx.clean_data.copy()

        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X = df[feature_cols].values
        y_raw = df[self._target_column].values

        # Encode: "setosa" → 0, "versicolor" → 1, "virginica" → 2
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)

        # Guardar el mapeo en el contexto para trazabilidad
        ctx.extra["label_mapping"] = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_))
        )

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = self._test_size,
            random_state = self._random_state,
            stratify     = y,   # garantiza proporción igual en train/test
        )

        # Scaler
        scaler       = StandardScaler()
        ctx.X_train  = scaler.fit_transform(X_train)
        ctx.X_test   = scaler.transform(X_test)
        ctx.y_train  = y_train
        ctx.y_test   = y_test

        # Guardar scaler para usarlo en inferencia si es necesario
        ctx.extra["scaler"]       = scaler
        ctx.extra["feature_cols"] = feature_cols

        logger.info(
            "iris_features_done",
            train_samples = len(X_train),
            test_samples  = len(X_test),
            label_mapping = ctx.extra["label_mapping"],
        )

        return ctx


# ---------------------------------------------------------------------------
# 4. Train — RandomForest con hiperparámetros configurables
# ---------------------------------------------------------------------------

class IrisTrainStep(TrainStep):
    """
    Entrena un RandomForestClassifier.
    Los hiperparámetros vienen del TrainingContext (tabla model_config en DB).
    """

    # Valores por defecto si no se pasan hiperparámetros
    DEFAULTS = {
        "n_estimators": 100,
        "max_depth":    None,
        "min_samples_split": 2,
        "random_state": 42,
    }

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        # Merge: defaults ← hiperparámetros de la config
        params = {**self.DEFAULTS, **ctx.hyperparams.get("random_forest", {})}

        logger.info("iris_train_start", params=params)

        model = RandomForestClassifier(**params)
        model.fit(ctx.X_train, ctx.y_train)

        ctx.trained_model = model

        logger.info(
            "iris_train_done",
            n_estimators  = model.n_estimators,
            n_features_in = model.n_features_in_,
            classes       = model.classes_.tolist(),
        )

        return ctx


# ---------------------------------------------------------------------------
# 5. Evaluate — métricas multiclase + umbral de calidad
# ---------------------------------------------------------------------------

class IrisEvaluateStep(EvaluateStep):
    """
    Extiende la evaluación base con:
    - reporte por clase
    - umbral de accuracy configurable
    - aborta el pipeline si no alcanza el mínimo
    """

    def __init__(self, min_accuracy: float = 0.90) -> None:
        super().__init__(min_accuracy=min_accuracy)

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        y_pred = ctx.trained_model.predict(ctx.X_test)

        accuracy = accuracy_score(ctx.y_test, y_pred)
        f1       = f1_score(ctx.y_test, y_pred, average="weighted")
        report   = classification_report(
            ctx.y_test, y_pred,
            target_names=["setosa", "versicolor", "virginica"],
            output_dict=True,
        )

        ctx.metrics = {
            "accuracy": round(accuracy, 4),
            "f1":       round(f1, 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"],    4),
                    "f1":        round(report[cls]["f1-score"],  4),
                }
                for cls in ["setosa", "versicolor", "virginica"]
            },
        }

        logger.info("iris_evaluate_done", metrics=ctx.metrics)

        # Aborta el pipeline si no alcanza el umbral
        if accuracy < self._min_accuracy:
            ctx.errors.append(
                f"Accuracy {accuracy:.4f} por debajo del mínimo {self._min_accuracy}. "
                "Modelo no registrado."
            )
            logger.warning(
                "iris_evaluate_failed",
                accuracy    = accuracy,
                min_required = self._min_accuracy,
            )

        return ctx


# ---------------------------------------------------------------------------
# 6. Register — guarda .pkl y actualiza el registry en memoria
# ---------------------------------------------------------------------------

class IrisRegisterStep(BaseTrainingStep):
    """
    Serializa el modelo entrenado a disco y lo registra en ModelRegistry.
    Solo se ejecuta si el pipeline no tiene errores (evaluate pasó).
    """

    def __init__(self, model_path: str = "models/") -> None:
        self._model_path = model_path

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        # Construir nombre del archivo con versión
        filename = f"{ctx.model_name}_v{ctx.version}.pkl"
        full_path = Path(self._model_path) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Serializar a disco
        joblib.dump(ctx.trained_model, full_path)
        logger.info("iris_model_saved", path=str(full_path))

        # Crear instancia con metadata completa
        from app.ml.models.iris_classifier import IrisClassifier

        metadata = ModelMetadata(
            name          = ctx.model_name,
            version       = ctx.version,
            feature_names = ctx.extra.get("feature_cols", []),
            target_name   = "species",
            metrics       = ctx.metrics,
            hyperparams   = ctx.hyperparams,
            extra         = {
                "label_mapping": ctx.extra.get("label_mapping", {}),
                "artifact_path": str(full_path),
            },
        )

        instance = IrisClassifier(metadata)
        instance.load(str(full_path))

        # Registrar — reemplaza versión anterior si existía
        model_registry.register(ctx.model_name, instance, allow_override=True)

        logger.info(
            "iris_registered",
            name    = ctx.model_name,
            version = ctx.version,
            metrics = ctx.metrics,
        )

        return ctx