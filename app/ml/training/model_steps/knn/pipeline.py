from app.ml.training.model_steps.knn.steps import AddDerivedFeatureStep, CleanColumnsStep, SaveModelStep, TransformColumnsStep, LoadDataStep
from app.ml.training.pipeline import TrainingPipeline

def build_knn_pipeline(
    model_path: str    = "models/"
) -> TrainingPipeline:

    return TrainingPipeline(steps=[
        LoadDataStep(),
        AddDerivedFeatureStep(),
        CleanColumnsStep(),
        TransformColumnsStep(),
        SaveModelStep()
    ])