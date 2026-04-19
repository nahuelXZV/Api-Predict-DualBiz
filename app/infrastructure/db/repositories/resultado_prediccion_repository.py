from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import ResultadoPrediccion


class ResultadoPrediccionRepository(RepositoryABC[ResultadoPrediccion]):
    def get_by_id(self, id: int) -> ResultadoPrediccion | None:
        return ResultadoPrediccion.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return ResultadoPrediccion.objects.filter(pk=id).exists()

    def list_all(self) -> list[ResultadoPrediccion]:
        return list(ResultadoPrediccion.objects.all())

    def save(self, entity: ResultadoPrediccion) -> None:
        entity.save()

    def update(self, id: int, entity: ResultadoPrediccion) -> None:
        ResultadoPrediccion.objects.filter(pk=id).update(
            cantidad_sugerida=entity.cantidad_sugerida,
            score=entity.score,
            posicion=entity.posicion,
        )

    def delete(self, id: int) -> None:
        ResultadoPrediccion.objects.filter(pk=id).delete()

    def list_by_lote(self, lote_id: int) -> list[ResultadoPrediccion]:
        return list(
            ResultadoPrediccion.objects.filter(lote_prediccion_id=lote_id).order_by(
                "posicion"
            )
        )

    def delete_by_lote(self, lote_id: int) -> None:
        ResultadoPrediccion.objects.filter(lote_prediccion_id=lote_id).delete()

    def bulk_create(self, resultados: list[ResultadoPrediccion]) -> None:
        ResultadoPrediccion.objects.bulk_create(resultados)
