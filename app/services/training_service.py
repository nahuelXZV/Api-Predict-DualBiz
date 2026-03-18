from app.ml.training.context import TrainingContext
from app.ml.training.model_steps.knn.pipeline import build_knn_pipeline

class TrainingService:

    def run(self, model_name: str, version: str, hyperparams: dict) -> dict:
        ctx = TrainingContext(
            model_name  = model_name,
            version     = version,
            hyperparams = hyperparams,
        )

        pipeline = build_knn_pipeline()
        result   = pipeline.run(ctx)

        return result.summary()