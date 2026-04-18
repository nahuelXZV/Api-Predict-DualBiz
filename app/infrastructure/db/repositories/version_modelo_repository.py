from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import VersionModelo
from app.domain.core.config import tz_now


class VersionModeloRepository(RepositoryABC[VersionModelo]):
    def get_by_id(self, id: int) -> VersionModelo | None:
        return VersionModelo.objects.filter(pk=id).first()

    def exists(self, id: int) -> bool:
        return VersionModelo.objects.filter(pk=id).exists()

    def list_all(self) -> list[VersionModelo]:
        return list(VersionModelo.objects.all())

    def save(self, entity: VersionModelo) -> None:
        entity.save()

    def update(self, id: int, entity: VersionModelo) -> None:
        VersionModelo.objects.filter(pk=id).update(
            nombre_modelo=entity.nombre_modelo,
            version=entity.version,
            entrenado_en=entity.entrenado_en,
            ruta_pkl=entity.ruta_pkl,
            tipo_fuente_datos=entity.tipo_fuente_datos,
            cantidad_clientes=entity.cantidad_clientes,
            cantidad_productos=entity.cantidad_productos,
            hiperparametros=entity.hiperparametros,
            activo=entity.activo,
        )

    def delete(self, id: int) -> None:
        VersionModelo.objects.filter(pk=id).delete()

    def deactivate_all(self, model_name: str) -> None:
        VersionModelo.objects.filter(nombre_modelo=model_name).update(activo=False)

    def create(self, **kwargs) -> VersionModelo:
        kwargs.setdefault("entrenado_en", tz_now())
        return VersionModelo.objects.create(**kwargs)
