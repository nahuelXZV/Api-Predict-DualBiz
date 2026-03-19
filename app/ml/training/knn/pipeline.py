from app.domain.ml.base_pipeline import BasePipeline
from app.ml.training.knn.steps import AddDerivedFeatureStep, CleanColumnsStep, LoadDataStep, RegistryModelStep, SaveModelStep, TransformColumnsStep

def build_knn_pipeline() -> BasePipeline:
    return BasePipeline(steps=[
        LoadDataStep(),
        AddDerivedFeatureStep(),
        CleanColumnsStep(),
        TransformColumnsStep(),
        SaveModelStep(),
        RegistryModelStep()
    ])