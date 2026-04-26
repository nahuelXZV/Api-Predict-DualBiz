from app.domain.models import ResultadoPrediccion
from app.infrastructure.db.repositories.resultado_prediccion_repository import (
    ResultadoPrediccionRepository,
)


class ResultadoPrediccionService:
    def __init__(self, repo: ResultadoPrediccionRepository) -> None:
        self._repo = repo

    def guardar_resultados(self, lote_id: int, resultados: list[dict]) -> None:
        objetos = [
            ResultadoPrediccion(
                lote_prediccion_id=lote_id,
                cliente_id=r["cliente_id"],
                producto_id=r["producto_id"],
                fuente=r["fuente"],
                cantidad_sugerida=r["cantidad_sugerida"],
                score=r["score"],
                posicion=r["posicion"],
                complementos=r.get("complementos", ""),
            )
            for r in resultados
        ]
        self._repo.bulk_create(objetos)

    def list_by_lote(self, lote_id: int) -> list:
        return self._repo.list_by_lote(lote_id)

    def reemplazar_resultados(self, lote_id: int, resultados: list[dict]) -> None:
        self._repo.delete_by_lote(lote_id)
        self.guardar_resultados(lote_id, resultados)


resultado_prediccion_service = ResultadoPrediccionService(
    repo=ResultadoPrediccionRepository()
)
