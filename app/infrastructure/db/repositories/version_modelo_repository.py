from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import VersionModelo
from app.domain.core.config import tz_now


class VersionModeloRepository(RepositoryABC[VersionModelo]):
    def get_by_id(self, id: int) -> VersionModelo | None:
        return VersionModelo.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return VersionModelo.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[VersionModelo]:
        return list(VersionModelo.objects.filter(eliminado=False))

    def save(self, entity: VersionModelo) -> None:
        entity.save()

    def update(self, id: int, entity: VersionModelo) -> None:
        VersionModelo.objects.filter(pk=id, eliminado=False).update(
            nombre_modelo=entity.nombre_modelo,
            version=entity.version,
            entrenado_en=entity.entrenado_en,
            ruta_pkl=entity.ruta_pkl,
            activo=entity.activo,
        )

    def delete(self, id: int) -> None:
        VersionModelo.objects.filter(pk=id, eliminado=False).update(eliminado=True)

    def get_activo(self, nombre_modelo: str) -> VersionModelo | None:
        return VersionModelo.objects.filter(
            nombre_modelo=nombre_modelo,
            activo=True,
            eliminado=False,
        ).first()

    def list_activos(self) -> list[VersionModelo]:
        return list(VersionModelo.objects.filter(activo=True, eliminado=False))

    def deactivate_all(self, model_name: str) -> None:
        VersionModelo.objects.filter(
            nombre_modelo=model_name,
            eliminado=False,
        ).update(activo=False)

    def create(self, **kwargs) -> VersionModelo:
        kwargs.setdefault("entrenado_en", tz_now())
        kwargs.setdefault("eliminado", False)
        return VersionModelo.objects.create(**kwargs)
