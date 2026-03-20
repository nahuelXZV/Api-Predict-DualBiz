from app.domain.ml.base_pipeline import BasePipeline
from app.ml.predict.knn.steps import PredictStep

def predict_knn_pipeline() -> BasePipeline:
    return BasePipeline(steps=[
        PredictStep()
    ])