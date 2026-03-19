from app.domain.ml.pipeline import BasePipeline
from app.ml.training.knn.steps import AddDerivedFeatureStep, CleanColumnsStep, SaveModelStep, TransformColumnsStep, LoadDataStep

def build_knn_pipeline() -> BasePipeline:
    return BasePipeline(steps=[
        LoadDataStep(),
        AddDerivedFeatureStep(),
        CleanColumnsStep(),
        TransformColumnsStep(),
        SaveModelStep()
    ])