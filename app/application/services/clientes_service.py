from django.db import transaction

from app.domain.models.clientes import Clientes
from app.infrastructure.db.repositories.clientes_repository import ClientesRepository


class ClientesService:
    def __init__(self, repo: ClientesRepository) -> None:
        self._repo = repo

    def obtener_cliente(self, cliente_id: int) -> Clientes | None:
        return self._repo.get_by_id(cliente_id)

    def listar_clientes(self) -> list[Clientes]:
        return self._repo.list_all()

    @transaction.atomic
    def crear_cliente(self, nombre_cliente: str, codigo_erp: str) -> Clientes:
        cliente = self._repo.create(
            nombre_cliente=nombre_cliente,
            codigo_erp=codigo_erp,
        )
        return cliente

    @transaction.atomic
    def actualizar_cliente(
        self, cliente_id: int, nombre_cliente: str, codigo_erp: str
    ) -> None:
        cliente = self._repo.get_by_id(cliente_id)
        if cliente is None:
            raise ValueError(f"Cliente {cliente_id} no encontrado.")

        cliente.nombre_cliente = nombre_cliente
        cliente.codigo_erp = codigo_erp
        self._repo.update(cliente_id, cliente)


cliente_service = ClientesService(repo=ClientesRepository())
