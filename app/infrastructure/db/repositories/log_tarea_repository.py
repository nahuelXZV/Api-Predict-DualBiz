from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.core.config import tz_now
from app.domain.models import LogTareaProgramada


class LogTareaRepository(RepositoryABC[LogTareaProgramada]):
    def get_by_id(self, id: int) -> LogTareaProgramada | None:
        return LogTareaProgramada.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return LogTareaProgramada.objects.filter(pk=id).exists()

    def list_all(self) -> list[LogTareaProgramada]:
        return list(LogTareaProgramada.objects.all())

    def save(self, entity: LogTareaProgramada) -> None:
        entity.save()

    def update(self, id: int, entity: LogTareaProgramada) -> None:
        LogTareaProgramada.objects.filter(pk=id).update(
            estado=entity.estado,
            duracion_segundos=entity.duracion_segundos,
            mensaje_error=entity.mensaje_error,
        )

    def delete(self, id: int) -> None:
        LogTareaProgramada.objects.filter(pk=id).delete()

    def list_by_ejecucion(self, ejecucion_id: int) -> list[LogTareaProgramada]:
        return list(
            LogTareaProgramada.objects.filter(
                ejecucion_tarea_programada_id=ejecucion_id
            ).order_by("orden_step")
        )

    def create_step(
        self,
        ejecucion_id: int,
        nombre_step: str,
        orden_step: int,
        estado: str,
        duracion_segundos: float | None = None,
        mensaje_error: str | None = None,
    ) -> LogTareaProgramada:
        return LogTareaProgramada.objects.create(
            ejecucion_tarea_programada_id=ejecucion_id,
            nombre_step=nombre_step,
            orden_step=orden_step,
            estado=estado,
            duracion_segundos=duracion_segundos,
            mensaje_error=mensaje_error,
            ejecutado_en=tz_now(),
        )
