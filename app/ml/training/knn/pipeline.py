from app.domain.ml.pipeline import TrainingPipeline
from app.ml.training.knn.steps import AddDerivedFeatureStep, CleanColumnsStep, SaveModelStep, TransformColumnsStep, LoadDataStep

def build_knn_pipeline() -> TrainingPipeline:
    return TrainingPipeline(steps=[
        LoadDataStep(),
        AddDerivedFeatureStep(),
        CleanColumnsStep(),
        TransformColumnsStep(),
        SaveModelStep()
    ])