from typing import Callable

_TRAIN_PIPELINES: dict[str, type] = {}


def register_pipeline(model_name: str) -> Callable:
    def decorator(cls: type) -> type:
        _TRAIN_PIPELINES[model_name] = cls
        return cls

    return decorator


def get_pipeline(model_name: str) -> type:
    pipeline = _TRAIN_PIPELINES.get(model_name)
    if not pipeline:
        raise ValueError(
            f"No hay pipeline registrado para '{model_name}'. "
            f"Disponibles: {list(_TRAIN_PIPELINES.keys())}"
        )
    return pipeline
