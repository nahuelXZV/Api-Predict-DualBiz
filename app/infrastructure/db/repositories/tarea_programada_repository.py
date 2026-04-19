from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import TareaProgramada


class TareaProgramadaRepository(RepositoryABC[TareaProgramada]):
    def get_by_id(self, id: int) -> TareaProgramada | None:
        return TareaProgramada.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return TareaProgramada.objects.filter(pk=id).exists()

    def list_all(self) -> list[TareaProgramada]:
        return list(TareaProgramada.objects.all())

    def list_active(self) -> list[TareaProgramada]:
        return list(TareaProgramada.objects.filter(activo=True))

    def save(self, entity: TareaProgramada) -> None:
        entity.save()

    def update(self, id: int, entity: TareaProgramada) -> None:
        TareaProgramada.objects.filter(pk=id).update(
            nombre=entity.nombre,
            tipo_job=entity.tipo_job,
            cron_schedule=entity.cron_schedule,
            activo=entity.activo,
        )

    def delete(self, id: int) -> None:
        TareaProgramada.objects.filter(pk=id).delete()
