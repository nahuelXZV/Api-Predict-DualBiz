from app.domain.models import MetricaModelo
from app.infrastructure.db.repositories.metrica_modelo_repository import (
    MetricaModeloRepository,
)


class MetricaModeloService:
    def __init__(self, repo: MetricaModeloRepository) -> None:
        self._repo = repo

    def guardar_metricas(self, version_modelo_id: int, metricas: dict) -> None:
        objetos = [
            MetricaModelo(
                version_modelo_id=version_modelo_id,
                nombre_metrica=nombre,
                valor_metrica=valor,
            )
            for nombre, valor in metricas.items()
        ]
        self._repo.bulk_create(objetos)

    def list_by_version(self, version_modelo_id: int) -> list:
        return self._repo.list_by_version(version_modelo_id)


metrica_modelo_service = MetricaModeloService(repo=MetricaModeloRepository())
