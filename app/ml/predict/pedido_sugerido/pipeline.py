from app.domain.ml.base_pipeline import BasePipeline
from app.ml.predict.pedido_sugerido.steps import (
    AprioriBuildCandidatesStep,
    AprioriRankAndPredictStep,
    BuildResponseStep,
<<<<<<< HEAD
    DestacadosStep,
=======
>>>>>>> bfaa3fb2cac645cc53a2b75ba8d9a7a20814fa99
    KnnBuildCandidatesStep,
    KnnFindNeighborsStep,
    KnnRankAndPredictStep,
    LoadModelStep,
<<<<<<< HEAD
    ParetoFilterStep,
=======
>>>>>>> bfaa3fb2cac645cc53a2b75ba8d9a7a20814fa99
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

<<<<<<< HEAD
    # Build response
    pipeline.add_step(ParetoFilterStep())
    pipeline.add_step(DestacadosStep())
=======
>>>>>>> bfaa3fb2cac645cc53a2b75ba8d9a7a20814fa99
    pipeline.add_step(BuildResponseStep())
    return pipeline
