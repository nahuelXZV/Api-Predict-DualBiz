from app.domain.ml.base_pipeline import BasePipeline
from app.ml.training.knn.steps import AddDerivedFeatureStep, CleanColumnsStep, LoadDataStep, RegistryModelStep, SaveModelStep, SegmentCustomersStep 

def build_knn_pipeline() -> BasePipeline:
    return BasePipeline(steps=[
        LoadDataStep(),
        AddDerivedFeatureStep(),
        CleanColumnsStep(),
        SegmentCustomersStep(),
        SaveModelStep(),
        RegistryModelStep()
    ])