from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models.clientes import Clientes


class ClientesRepository(RepositoryABC[Clientes]):
    def get_by_id(self, id: int) -> Clientes | None:
        return Clientes.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return Clientes.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[Clientes]:
        return list(Clientes.objects.filter(eliminado=False))

    def save(self, entity: Clientes) -> None:
        entity.save()

    def update(self, id: int, entity: Clientes) -> None:
        Clientes.objects.filter(pk=id, eliminado=False).update(
            nombre_cliente=entity.nombre_cliente,
            codigo_erp=entity.codigo_erp,
        )

    def delete(self, id: int) -> None:
        Clientes.objects.filter(pk=id, eliminado=False).update(eliminado=True)

    def create(self, **kwargs) -> Clientes:
        kwargs.setdefault("eliminado", False)
        return Clientes.objects.create(**kwargs)
