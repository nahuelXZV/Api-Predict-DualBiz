from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.core.config import tz_now
from app.domain.models import LotePrediccion


class LotePrediccionRepository(RepositoryABC[LotePrediccion]):
    def get_by_id(self, id: int) -> LotePrediccion | None:
        return LotePrediccion.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return LotePrediccion.objects.filter(pk=id).exists()

    def list_all(self) -> list[LotePrediccion]:
        return list(LotePrediccion.objects.all())

    def save(self, entity: LotePrediccion) -> None:
        entity.save()

    def update(self, id: int, entity: LotePrediccion) -> None:
        LotePrediccion.objects.filter(pk=id).update(
            estado=entity.estado,
            cantidad_predicciones=entity.cantidad_predicciones,
            parametros=entity.parametros,
        )

    def delete(self, id: int) -> None:
        LotePrediccion.objects.filter(pk=id).delete()

    def get_by_modelo(self, nombre_modelo: str) -> LotePrediccion | None:
        return LotePrediccion.objects.filter(nombre_modelo=nombre_modelo).first()

    def create(self, **kwargs) -> LotePrediccion:
        kwargs.setdefault("generado_en", tz_now())
        return LotePrediccion.objects.create(**kwargs)

    def marcar_completado(self, id: int, cantidad_predicciones: int) -> None:
        LotePrediccion.objects.filter(pk=id).update(
            estado="completado",
            cantidad_predicciones=cantidad_predicciones,
        )

    def marcar_fallido(self, id: int) -> None:
        LotePrediccion.objects.filter(pk=id).update(estado="fallido")
