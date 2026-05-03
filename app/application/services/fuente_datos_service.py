from app.domain.models.fuente_datos import FuenteDatos
from app.infrastructure.db.repositories.fuente_datos_repository import (
    FuenteDatosRepository,
)
from app.infrastructure.db.repositories.fuente_datos_parametros_repository import (
    FuenteDatosParametrosRepository,
)


class FuenteDatosService:
    def __init__(
        self,
        repo: FuenteDatosRepository,
        repoParametros: FuenteDatosParametrosRepository,
    ) -> None:
        self._repo = repo
        self._repoParametros = repoParametros

    def obtener_fuente_datos(self, fuente_id: int) -> FuenteDatos | None:
        return self._repo.get_by_id(fuente_id)

    def listar_fuente_datos(self) -> list[FuenteDatos]:
        return self._repo.list_all()

    def obtener_parametros_fuente_datos(self, fuente_id: int) -> dict[str, str]:
        return self._repoParametros.get_by_fuente(fuente_id)


fuente_datos_service = FuenteDatosService(
    repo=FuenteDatosRepository(), repoParametros=FuenteDatosParametrosRepository()
)
