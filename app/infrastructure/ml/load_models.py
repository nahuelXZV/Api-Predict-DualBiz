from pathlib import Path

from app.domain.core.config import settings
from app.domain.core.logging import logger
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.infrastructure.ml.models.pedido_sugerido_model import PedidoSugeridoModel

_MODELS_TO_LOAD = [
    "pedido_sugerido",
]


def _find_latest_pkl(model_name: str) -> Path | None:
    """
    Busca en settings.path_models el archivo .pkl más reciente
    que coincida con el patrón modelo_{model_name}_*.pkl.
    """
    base = Path(settings.path_models)
    matches = sorted(
        base.glob(f"modelo_{model_name}_*.pkl"), key=lambda p: p.stat().st_mtime
    )
    return matches[-1] if matches else None


def load_initial_models() -> None:
    """
    Carga y registra todos los modelos definidos en _MODELS_TO_LOAD
    al inicio de la aplicación. Se ejecuta en AppConfig.ready().
    """
    for model_name in _MODELS_TO_LOAD:
        path = _find_latest_pkl(model_name)

        if path is None:
            logger.warning(
                "model_file_not_found", model=model_name, dir=settings.path_models
            )
            continue

        version = path.stem.replace(f"modelo_{model_name}_", "")

        try:
            meta = ModelMetadata(name=model_name, version=version, path_model=str(path))
            model = PedidoSugeridoModel(metadata=meta)
            model.load(str(path))
            model_registry.register(name=model_name, model=model)
            logger.info(
                "model_loaded_at_startup",
                model=model_name,
                version=version,
                path=str(path),
            )
        except Exception as e:
            logger.error(
                "model_load_failed", model=model_name, path=str(path), error=str(e)
            )
