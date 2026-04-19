from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.core.config import tz_now
from app.domain.models import EjecucionTareaProgramada
from app.domain.utils.enums import EstadoEjecucion


class EjecucionTareaRepository(RepositoryABC[EjecucionTareaProgramada]):
    def get_by_id(self, id: int) -> EjecucionTareaProgramada | None:
        return EjecucionTareaProgramada.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return EjecucionTareaProgramada.objects.filter(pk=id).exists()

    def list_all(self) -> list[EjecucionTareaProgramada]:
        return list(EjecucionTareaProgramada.objects.all())

    def save(self, entity: EjecucionTareaProgramada) -> None:
        entity.save()

    def update(self, id: int, entity: EjecucionTareaProgramada) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id).update(
            estado=entity.estado,
            finalizado_en=entity.finalizado_en,
            mensaje_error=entity.mensaje_error,
        )

    def delete(self, id: int) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id).delete()

    def create_inicio(
        self, tarea_id: int, disparado_por: str
    ) -> EjecucionTareaProgramada:
        return EjecucionTareaProgramada.objects.create(
            tarea_programada_id=tarea_id,
            disparado_por=disparado_por,
            estado=EstadoEjecucion.EJECUTANDO.value,
            iniciado_en=tz_now(),
        )

    def marcar_exitosa(self, id: int) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id).update(
            estado=EstadoEjecucion.EXITOSO.value,
            finalizado_en=tz_now(),
        )

    def marcar_fallida(self, id: int, mensaje_error: str) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id).update(
            estado=EstadoEjecucion.FALLIDO.value,
            finalizado_en=tz_now(),
            mensaje_error=mensaje_error,
        )
