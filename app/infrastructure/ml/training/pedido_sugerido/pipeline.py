from app.domain.ml.base_pipeline import BasePipeline
from app.domain.ml.data_source import DataSource
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


def build_pedido_sugerido_pipeline(data_source: DataSource) -> BasePipeline:
    pipeline = BasePipeline()

    pipeline.add_step(LoadDataStep(data_source))
    pipeline.add_step(EdaCleanDataStep())
    pipeline.add_step(CalculoAtributosDerivadosStep())
    pipeline.add_step(ClusteringKMeansStep())
    pipeline.add_step(VecinosCercanosKnnStep())
    pipeline.add_step(ConjuntoReglasAprioriStep())
    pipeline.add_step(PrepareDataArbolesStep())
    # pipeline.add_step(EnsembleArbolesXGBoostStep())
    pipeline.add_step(EnsembleArbolesRandomForestStep())
    pipeline.add_step(SaveModelStep())
    pipeline.add_step(RegistryModelStep())

    return pipeline
