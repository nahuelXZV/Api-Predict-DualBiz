from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.core.config import tz_now
from app.domain.models.clientes import Clientes

class ClientesRepository(RepositoryABC[Clientes]):
    def get_by_id(self, id: int) -> Clientes | None:
        return Clientes.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return Clientes.objects.filter(pk=id).exists()

    def list_all(self) -> list[Clientes]:
        return list(Clientes.objects.all())

    def save(self, entity: Clientes) -> None:
        entity.save()

    def update(self, id: int, entity: Clientes) -> None:
        Clientes.objects.filter(pk=id).update(
            nombre_cliente=entity.nombre_cliente,
            codigo_erp=entity.codigo_erp,
            registrado_en=tz_now(),
        )

    def delete(self, id: int) -> None:
        Clientes.objects.filter(pk=id).delete()

    def create(self, **kwargs) -> Clientes:
        kwargs.setdefault("registrado_en", tz_now())
        return Clientes.objects.create(**kwargs)
