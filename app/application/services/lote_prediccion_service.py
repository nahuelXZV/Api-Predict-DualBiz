from app.infrastructure.db.repositories.lote_prediccion_repository import (
    LotePrediccionRepository,
)
from app.infrastructure.db.repositories.version_modelo_repository import (
    VersionModeloRepository,
)


class LotePrediccionService:
    def __init__(
        self,
        lote_repo: LotePrediccionRepository,
        version_repo: VersionModeloRepository,
    ) -> None:
        self._lote_repo = lote_repo
        self._version_repo = version_repo

    def iniciar_lote(self, nombre_modelo: str, parametros: dict) -> int:
        version = self._version_repo.get_activo(nombre_modelo)
        if version is None:
            raise ValueError(f"No hay versión activa para el modelo '{nombre_modelo}'.")

        lote = self._lote_repo.create(
            nombre_modelo=nombre_modelo,
            version_modelo=version,
            parametros=parametros,
            estado="generando",
        )
        return lote.id

    def completar_lote(self, lote_id: int, cantidad_predicciones: int) -> None:
        self._lote_repo.marcar_completado(lote_id, cantidad_predicciones)

    def fallar_lote(self, lote_id: int) -> None:
        self._lote_repo.marcar_fallido(lote_id)

    def get_lote_activo(self, nombre_modelo: str):
        return self._lote_repo.get_by_modelo(nombre_modelo)


lote_prediccion_service = LotePrediccionService(
    lote_repo=LotePrediccionRepository(),
    version_repo=VersionModeloRepository(),
)
