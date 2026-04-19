from app.infrastructure.db.repositories.log_tarea_repository import LogTareaRepository


class LogTareaService:
    def __init__(self, repo: LogTareaRepository) -> None:
        self._repo = repo

    def registrar_step(
        self,
        ejecucion_id: int,
        nombre_step: str,
        orden_step: int,
        estado: str,
        duracion_segundos: float | None = None,
        mensaje_error: str | None = None,
    ) -> None:
        self._repo.create_step(
            ejecucion_id=ejecucion_id,
            nombre_step=nombre_step,
            orden_step=orden_step,
            estado=estado,
            duracion_segundos=duracion_segundos,
            mensaje_error=mensaje_error,
        )

    def list_by_ejecucion(self, ejecucion_id: int) -> list:
        return self._repo.list_by_ejecucion(ejecucion_id)


log_tarea_service = LogTareaService(repo=LogTareaRepository())
