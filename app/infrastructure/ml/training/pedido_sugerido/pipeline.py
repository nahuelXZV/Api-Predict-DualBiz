from app.domain.ml.abstractions.pipeline_base import PipelineBase
from app.domain.ml.abstractions.data_source_abc import DataSourceABC
from app.infrastructure.ml.pipeline_registry import register_pipeline
from app.infrastructure.ml.training.pedido_sugerido.steps import (
    CalculoAtributosDerivadosStep,
    ClusteringKMeansStep,
    ConjuntoReglasAprioriStep,
    EdaCleanDataStep,
    EnsembleArbolesRandomForestStep,
    LoadDataStep,
    PrepareDataArbolesStep,
    RegistryModelStep,
    SaveModelStep,
    VecinosCercanosKnnStep,
)


@register_pipeline("pedido_sugerido")
class PedidoSugeridoPipeline(PipelineBase):
    def __init__(self, data_source: DataSourceABC) -> None:
        super().__init__()
        self.add_step(LoadDataStep(data_source))
        self.add_step(EdaCleanDataStep())
        self.add_step(CalculoAtributosDerivadosStep())
        self.add_step(ClusteringKMeansStep())
        self.add_step(VecinosCercanosKnnStep())
        self.add_step(ConjuntoReglasAprioriStep())
        self.add_step(PrepareDataArbolesStep())
        self.add_step(EnsembleArbolesRandomForestStep())
        self.add_step(SaveModelStep())
        self.add_step(RegistryModelStep())
