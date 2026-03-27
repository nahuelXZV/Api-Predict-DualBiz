from app.domain.ml.base_pipeline import BasePipeline
from app.ml.training.pedido_sugerido.steps import (
    CalculatedFeatureStep,
    CleanDataStep,
    KnnStep,
    LoadDataStep,
    PrepareDataXGBStep,
    RegistryModelStep,
    SaveModelStep,
    KMeansStep,
    TrainCantidadStep,
)


def build_pedido_sugerido_pipeline() -> BasePipeline:
    pipeline = BasePipeline()

    pipeline.add_step(LoadDataStep())
    pipeline.add_step(CleanDataStep())
    pipeline.add_step(CalculatedFeatureStep())
    pipeline.add_step(KMeansStep())
    pipeline.add_step(KnnStep())
    pipeline.add_step(PrepareDataXGBStep())
    pipeline.add_step(TrainCantidadStep())
    pipeline.add_step(SaveModelStep())
    pipeline.add_step(RegistryModelStep())

    return pipeline
