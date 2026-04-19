from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import MetricaModelo


class MetricaModeloRepository(RepositoryABC[MetricaModelo]):
    def get_by_id(self, id: int) -> MetricaModelo | None:
        return MetricaModelo.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return MetricaModelo.objects.filter(pk=id).exists()

    def list_all(self) -> list[MetricaModelo]:
        return list(MetricaModelo.objects.all())

    def save(self, entity: MetricaModelo) -> None:
        entity.save()

    def update(self, id: int, entity: MetricaModelo) -> None:
        MetricaModelo.objects.filter(pk=id).update(
            nombre_metrica=entity.nombre_metrica,
            valor_metrica=entity.valor_metrica,
            split=entity.split,
        )

    def delete(self, id: int) -> None:
        MetricaModelo.objects.filter(pk=id).delete()

    def list_by_version(self, version_modelo_id: int) -> list[MetricaModelo]:
        return list(MetricaModelo.objects.filter(version_modelo_id=version_modelo_id))

    def create(self, **kwargs) -> MetricaModelo:
        return MetricaModelo.objects.create(**kwargs)

    def bulk_create(self, metricas: list[MetricaModelo]) -> None:
        MetricaModelo.objects.bulk_create(metricas)
