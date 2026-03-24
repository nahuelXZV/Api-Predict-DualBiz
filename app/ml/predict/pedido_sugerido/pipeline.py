from app.domain.ml.base_pipeline import BasePipeline
from app.ml.predict.pedido_sugerido.steps import (
    BuildCandidatesStep,
    BuildFeatureMatrixStep,
    FindNeighborsStep,
    LoadModelStep,
    PredictStep,
    ValidateClienteStep,
)


def predict_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()
    pipeline.add_step(LoadModelStep())
    pipeline.add_step(ValidateClienteStep())
    pipeline.add_step(FindNeighborsStep())
    pipeline.add_step(BuildCandidatesStep())
    pipeline.add_step(BuildFeatureMatrixStep())
    pipeline.add_step(PredictStep())
    return pipeline


