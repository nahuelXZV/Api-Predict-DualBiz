from typing import Callable

from app.domain.ml.abstractions.pipeline_base import PipelineBase
from app.domain.ml.abstractions.training_pipeline_base import TrainingPipelineBase

_PIPELINES: dict[str, type[PipelineBase]] = {}


def register_pipeline(
    model_name: str,
) -> Callable[[type[PipelineBase]], type[PipelineBase]]:
    def decorator(cls: type[PipelineBase]) -> type[PipelineBase]:
        if not issubclass(cls, PipelineBase):
            raise TypeError(
                f"'{cls.__name__}' debe heredar de PipelineBase para ser registrado."
            )
        _PIPELINES[model_name] = cls
        return cls

    return decorator


def get_pipeline(model_name: str) -> type[PipelineBase]:
    pipeline = _PIPELINES.get(model_name)
    if not pipeline:
        raise ValueError(
            f"No hay pipeline registrado para '{model_name}'. "
            f"Disponibles: {list(_PIPELINES.keys())}"
        )
    return pipeline


def get_training_pipeline(model_name: str) -> type[TrainingPipelineBase]:
    pipeline = get_pipeline(model_name)
    if not issubclass(pipeline, TrainingPipelineBase):
        raise TypeError(f"El pipeline '{model_name}' no es un TrainingPipelineBase.")
    return pipeline
