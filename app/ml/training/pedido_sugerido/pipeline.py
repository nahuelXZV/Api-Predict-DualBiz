from app.domain.ml.base_pipeline import BasePipeline
from app.ml.training.pedido_sugerido.steps import (
    CalculoAtributosDerivadosStep,
    ClusteringKMeansStep,
    ConjuntoReglasAprioriStep,
    EdaCleanDataStep,
    EnsembleArbolesRandomForestStep,
    # EnsembleArbolesXGBoostStep,
    LoadDataStep,
    PrepareDataXGBStep,
    RegistryModelStep,
    SaveModelStep,
    VecinosCercanosKnnStep,
)


def build_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()

    pipeline.add_step(LoadDataStep())
    pipeline.add_step(EdaCleanDataStep())
    pipeline.add_step(CalculoAtributosDerivadosStep())
    pipeline.add_step(ClusteringKMeansStep())
    pipeline.add_step(VecinosCercanosKnnStep())
    pipeline.add_step(ConjuntoReglasAprioriStep())
    pipeline.add_step(PrepareDataXGBStep())
    # pipeline.add_step(EnsembleArbolesXGBoostStep())
    pipeline.add_step(EnsembleArbolesRandomForestStep())
    pipeline.add_step(SaveModelStep())
    pipeline.add_step(RegistryModelStep())

    return pipeline
