from app.domain.abstractions.pipeline_base import TrainingPipelineBase
from app.domain.abstractions.data_source_abc import DataSourceABC
from app.application.ml.pipeline_registry import register_pipeline
from app.application.ml.pipelines.training.pedido_sugerido.steps import (
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
class PedidoSugeridoPipeline(TrainingPipelineBase):
    def __init__(self) -> None:
        super().__init__()
        self._data_source: DataSourceABC | None = None

    def set_datasource(self, data_source: DataSourceABC) -> None:
        self._data_source = data_source

    def build_steps(self) -> None:
        if self._data_source is None:
            raise ValueError("Debe llamar a set_datasource antes de set_steps")
        self._steps.clear()
        self.add_step(LoadDataStep(self._data_source))
        self.add_step(EdaCleanDataStep())
        self.add_step(CalculoAtributosDerivadosStep())
        self.add_step(ClusteringKMeansStep())
        self.add_step(VecinosCercanosKnnStep())
        self.add_step(ConjuntoReglasAprioriStep())
        self.add_step(PrepareDataArbolesStep())
        self.add_step(EnsembleArbolesRandomForestStep())
        self.add_step(SaveModelStep())
        self.add_step(RegistryModelStep())
