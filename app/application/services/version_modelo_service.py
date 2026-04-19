from django.db import transaction

from app.domain.ml.pipeline_context import TrainingContext
from app.infrastructure.db.repositories.version_modelo_repository import (
    VersionModeloRepository,
)


class VersionModeloService:
    def __init__(self, repo: VersionModeloRepository) -> None:
        self._repo = repo

    @transaction.atomic
    def save_new_version(
        self, ctx: TrainingContext, path_model: str, parameters: dict
    ) -> None:
        self._repo.deactivate_all(ctx.model_name)
        self._repo.create(
            nombre_modelo=ctx.model_name,
            version=ctx.version,
            ruta_pkl=path_model,
            tipo_fuente_datos="historial_ventas",
            parametros=parameters,
            ejecucion_tarea_programada_id=ctx.ejecucion_id,
            activo=True,
        )

    def get_version_activa(self, model_name: str):
        return self._repo.get_activo(model_name)

version_modelo_service = VersionModeloService(repo=VersionModeloRepository())
