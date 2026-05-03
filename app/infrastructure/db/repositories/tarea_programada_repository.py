from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import TareaProgramada


class TareaProgramadaRepository(RepositoryABC[TareaProgramada]):
    def get_by_id(self, id: int) -> TareaProgramada | None:
        return TareaProgramada.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return TareaProgramada.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[TareaProgramada]:
        return list(TareaProgramada.objects.filter(eliminado=False))

    def list_active(self) -> list[TareaProgramada]:
        return list(TareaProgramada.objects.filter(activo=True, eliminado=False))

    def save(self, entity: TareaProgramada) -> None:
        entity.save()

    def update(self, id: int, entity: TareaProgramada) -> None:
        TareaProgramada.objects.filter(pk=id, eliminado=False).update(
            nombre=entity.nombre,
            tipo_job=entity.tipo_job,
            cron_schedule=entity.cron_schedule,
            activo=entity.activo,
        )

    def delete(self, id: int) -> None:
        TareaProgramada.objects.filter(pk=id, eliminado=False).update(eliminado=True)
