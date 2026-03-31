from app.domain.ml.base_pipeline import BasePipeline
from app.infrastructure.ml.predict.pedido_sugerido.steps import (
    AprioriBuildCandidatesStep,
    AprioriRankAndPredictStep,
    BuildResponseStep,
    DestacadosStep,
    KnnBuildCandidatesStep,
    KnnFindNeighborsStep,
    KnnRankAndPredictStep,
    LoadModelStep,
    ParetoFilterStep,
    ValidateClienteStep,
)


def predict_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()
    pipeline.add_step(LoadModelStep())
    pipeline.add_step(ValidateClienteStep())

    # KNN + XGBoost
    pipeline.add_step(KnnFindNeighborsStep())
    pipeline.add_step(KnnBuildCandidatesStep())
    pipeline.add_step(KnnRankAndPredictStep())
    # Apriori + XGBoost
    pipeline.add_step(AprioriBuildCandidatesStep())
    pipeline.add_step(AprioriRankAndPredictStep())

    # Build response
    pipeline.add_step(ParetoFilterStep())
    pipeline.add_step(DestacadosStep())
    pipeline.add_step(BuildResponseStep())
    return pipeline
