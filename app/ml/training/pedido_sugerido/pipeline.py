from app.domain.ml.base_pipeline import BasePipeline
from app.ml.training.pedido_sugerido.steps import (
    CalculoAtributosDerivadosStep,
    EDA_CleanDataStep,
    VecinosCercanos_KnnStep,
    LoadDataStep,
    PrepareDataXGBStep,
    RegistryModelStep,
    SaveModelStep,
    Clustering_KMeansStep,
    EnsembleArboles_XGBoostStep,
    ConjuntoReglasApriori_Step,
)


def build_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()

    pipeline.add_step(LoadDataStep())
    pipeline.add_step(EDA_CleanDataStep())
    pipeline.add_step(CalculoAtributosDerivadosStep())
    pipeline.add_step(Clustering_KMeansStep())
    pipeline.add_step(VecinosCercanos_KnnStep())
    pipeline.add_step(ConjuntoReglasApriori_Step())
    pipeline.add_step(PrepareDataXGBStep())
    pipeline.add_step(EnsembleArboles_XGBoostStep())
    pipeline.add_step(SaveModelStep())
    pipeline.add_step(RegistryModelStep())

    return pipeline
