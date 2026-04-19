from app.domain.core.logging import logger
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.application.ml.predictors.pedido_sugerido import PedidoSugerido
from app.infrastructure.db.repositories.version_modelo_repository import VersionModeloRepository

_repo = VersionModeloRepository()


def load_initial_models() -> None:
    versiones = _repo.list_activos()

    if not versiones:
        logger.warning("no_active_model_versions_found")
        return

    for vm in versiones:
        try:
            meta = ModelMetadata(
                name=vm.nombre_modelo,
                version=vm.version,
                path_model=vm.ruta_pkl,
            )
            model = PedidoSugerido(metadata=meta)
            model.load(vm.ruta_pkl)
            model_registry.register(name=vm.nombre_modelo, model=model)
            logger.info(
                "model_loaded_at_startup",
                model=vm.nombre_modelo,
                version=vm.version,
                path=vm.ruta_pkl,
            )
        except Exception as e:
            logger.error(
                "model_load_failed",
                model=vm.nombre_modelo,
                path=vm.ruta_pkl,
                error=str(e),
            )
