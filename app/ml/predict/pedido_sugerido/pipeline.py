from app.domain.ml.base_pipeline import BasePipeline
from app.ml.predict.pedido_sugerido.steps import (
    BuildCandidatesStep,
    BuildFeaturesStep,
    FindNeighborsStep,
    LoadModelStep,
    RankAndPredictStep,
    ValidateClienteStep,
)


def predict_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()
    pipeline.add_step(LoadModelStep())
    pipeline.add_step(ValidateClienteStep())
    pipeline.add_step(FindNeighborsStep())
    pipeline.add_step(BuildCandidatesStep())
    pipeline.add_step(BuildFeaturesStep())
    pipeline.add_step(RankAndPredictStep())
    return pipeline
