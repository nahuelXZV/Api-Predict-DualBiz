from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.ml.pipeline_context import TrainingContext
from app.domain.models.version_modelo import VersionModelo
from app.infrastructure.db.repositories.version_modelo_repository import (
    VersionModeloRepository,
)


class VersionModeloService:
    def __init__(self, repo: RepositoryABC[VersionModelo]) -> None:
        self._repo = repo
    
    def save_new_version(self, ctx: TrainingContext, path_model: str) -> None:
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
            hiperparametros=ctx.hyperparams,
        )


version_modelo_service = VersionModeloService(repo=VersionModeloRepository())
