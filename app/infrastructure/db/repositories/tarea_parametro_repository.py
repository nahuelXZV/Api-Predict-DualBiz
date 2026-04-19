from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import TareaParametro


class TareaParametroRepository(RepositoryABC[TareaParametro]):
    def get_by_id(self, id: int) -> TareaParametro | None:
        return TareaParametro.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return TareaParametro.objects.filter(pk=id).exists()

    def list_all(self) -> list[TareaParametro]:
        return list(TareaParametro.objects.all())

    def save(self, entity: TareaParametro) -> None:
        entity.save()

    def update(self, id: int, entity: TareaParametro) -> None:
        TareaParametro.objects.filter(pk=id).update(
            clave=entity.clave,
            valor=entity.valor,
        )

    def delete(self, id: int) -> None:
        TareaParametro.objects.filter(pk=id).delete()

    def get_by_tarea(self, tarea_id: int) -> dict:
        return {
            p.clave: p.valor
            for p in TareaParametro.objects.filter(tarea_programada_id=tarea_id)
        }

    def set_param(self, tarea_id: int, clave: str, valor: str) -> None:
        TareaParametro.objects.update_or_create(
            tarea_programada_id=tarea_id,
            clave=clave,
            defaults={"valor": valor},
        )

    def delete_by_tarea(self, tarea_id: int) -> None:
        TareaParametro.objects.filter(tarea_programada_id=tarea_id).delete()
