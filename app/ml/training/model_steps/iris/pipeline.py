from app.ml.training.pipeline import TrainingPipeline
from app.ml.training.model_steps.iris.steps import (
    LoadDataStep,
)

def build_iris_pipeline(
    model_path: str    = "models/",
    min_accuracy: float = 0.90,
) -> TrainingPipeline:
    """
    Construye el pipeline completo para entrenar IrisClassifier.

    Pasos:
      1. IngestStep        — lee iris.csv (compartido, no cambia)
      2. IrisCleanStep     — limpieza base + validaciones específicas de Iris
      3. IrisFeatureStep   — encode target + split estratificado + StandardScaler
      4. IrisTrainStep     — RandomForestClassifier con hiperparámetros configurables
      5. IrisEvaluateStep  — accuracy, f1 por clase, umbral de calidad
      6. IrisRegisterStep  — guarda .pkl y actualiza ModelRegistry

    Args:
        model_path:   carpeta donde guardar el artefacto .pkl
        min_accuracy: accuracy mínima para aprobar el modelo (default 0.90)

    Returns:
        TrainingPipeline listo para llamar .run(ctx)

    Uso en TrainingJob:
        pipeline = build_iris_pipeline()
        ctx = TrainingContext(model_name="iris", version="20240101", ...)
        result = pipeline.run(ctx)
    """
    return TrainingPipeline(steps=[
        LoadDataStep(),
    ])