from app.domain.ml.abstractions.pipeline_base import PredictionPipelineBase
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


class PedidoSugeridoPredictPipeline(PredictionPipelineBase):
    def __init__(self) -> None:
        super().__init__()

    def build_steps(self) -> None:
        self._steps.clear()
        self.add_step(LoadModelStep())
        self.add_step(ValidateClienteStep())

        # KNN + XGBoost
        self.add_step(KnnFindNeighborsStep())
        self.add_step(KnnBuildCandidatesStep())
        self.add_step(KnnRankAndPredictStep())
        # Apriori + XGBoost
        self.add_step(AprioriBuildCandidatesStep())
        self.add_step(AprioriRankAndPredictStep())

        # Build response
        self.add_step(ParetoFilterStep())
        self.add_step(DestacadosStep())
        self.add_step(BuildResponseStep())
