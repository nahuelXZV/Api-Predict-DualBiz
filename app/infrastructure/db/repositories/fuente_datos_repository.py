from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import FuenteDatos


class FuenteDatosRepository(RepositoryABC[FuenteDatos]):
    def get_by_id(self, id: int) -> FuenteDatos | None:
        return FuenteDatos.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return FuenteDatos.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[FuenteDatos]:
        return list(FuenteDatos.objects.filter(eliminado=False))

    def save(self, entity: FuenteDatos) -> None:
        entity.save()

    def update(self, id: int, entity: FuenteDatos) -> None:
        FuenteDatos.objects.filter(pk=id, eliminado=False).update(
            nombre=entity.nombre,
            descripcion=entity.descripcion,
            tipo=entity.tipo,
        )

    def delete(self, id: int) -> None:
        FuenteDatos.objects.filter(pk=id, eliminado=False).update(eliminado=True)

    def create(self, **kwargs) -> FuenteDatos:
        kwargs.setdefault("eliminado", False)
        return FuenteDatos.objects.create(**kwargs)

    def get_by_nombre(self, nombre: str) -> FuenteDatos | None:
        return FuenteDatos.objects.filter(nombre=nombre, eliminado=False).first()

    def list_by_tipo(self, tipo: str) -> list[FuenteDatos]:
        return list(FuenteDatos.objects.filter(tipo=tipo, eliminado=False))
