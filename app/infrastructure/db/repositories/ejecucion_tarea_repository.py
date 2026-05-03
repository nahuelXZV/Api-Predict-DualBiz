from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.core.config import tz_now
from app.domain.models import EjecucionTareaProgramada
from app.domain.utils.enums import EstadoEjecucion


class EjecucionTareaRepository(RepositoryABC[EjecucionTareaProgramada]):
    def get_by_id(self, id: int) -> EjecucionTareaProgramada | None:
        return EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[EjecucionTareaProgramada]:
        return list(EjecucionTareaProgramada.objects.filter(eliminado=False))

    def save(self, entity: EjecucionTareaProgramada) -> None:
        entity.save()

    def update(self, id: int, entity: EjecucionTareaProgramada) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).update(
            estado=entity.estado,
            finalizado_en=entity.finalizado_en,
            mensaje_error=entity.mensaje_error,
        )

    def delete(self, id: int) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).update(
            eliminado=True
        )

    def create_inicio(
        self,
        tarea_id: int,
        disparado_por: str,
        numero_intento: int = 1,
        ejecucion_original_id: int | None = None,
    ) -> EjecucionTareaProgramada:
        return EjecucionTareaProgramada.objects.create(
            tarea_programada_id=tarea_id,
            disparado_por=disparado_por,
            estado=EstadoEjecucion.EJECUTANDO.value,
            numero_intento=numero_intento,
            ejecucion_original_id=ejecucion_original_id,
            iniciado_en=tz_now(),
            eliminado=False,
        )

    def marcar_exitosa(self, id: int) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).update(
            estado=EstadoEjecucion.EXITOSO.value,
            finalizado_en=tz_now(),
        )

    def marcar_fallida(self, id: int, mensaje_error: str) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).update(
            estado=EstadoEjecucion.FALLIDO.value,
            finalizado_en=tz_now(),
            mensaje_error=mensaje_error,
        )

    def marcar_pendiente_reintento(self, id: int) -> None:
        EjecucionTareaProgramada.objects.filter(pk=id, eliminado=False).update(
            estado=EstadoEjecucion.PENDIENTE_REINTENTO.value,
            finalizado_en=tz_now(),
        )
