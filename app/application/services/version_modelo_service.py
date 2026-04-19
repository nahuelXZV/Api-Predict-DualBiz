from app.domain.ml.pipeline_context import TrainingContext
from app.infrastructure.db.repositories.version_modelo_repository import (
    VersionModeloRepository,
)


class VersionModeloService:
    def __init__(self, repo: VersionModeloRepository) -> None:
        self._repo = repo

    def save_new_version(
        self, ctx: TrainingContext, path_model: str, hyperparams: dict
    ) -> None:
        cantidad_clientes = 0
        cantidad_productos = 0
        if ctx.clean_data is not None:
            cantidad_clientes = ctx.clean_data["cliente_id"].nunique()
            cantidad_productos = ctx.clean_data["producto_id"].nunique()

        self._repo.deactivate_all(ctx.model_name)
        self._repo.create(
            nombre_modelo=ctx.model_name,
            version=ctx.version,
            ruta_pkl=path_model,
            tipo_fuente_datos="historial_ventas",
            cantidad_clientes=cantidad_clientes,
            cantidad_productos=cantidad_productos,
            hiperparametros=hyperparams,
            activo=True,
        )

    def get_version_activa(self, model_name: str):
        return self._repo.get_activo(model_name)

version_modelo_service = VersionModeloService(repo=VersionModeloRepository())
