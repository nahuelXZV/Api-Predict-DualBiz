from app.domain.abstractions.repository_abc import RepositoryABC
from app.domain.models import FuenteDatosParametros


class FuenteDatosParametrosRepository(RepositoryABC[FuenteDatosParametros]):
    def get_by_id(self, id: int) -> FuenteDatosParametros | None:
        return FuenteDatosParametros.objects.filter(pk=id, eliminado=False).first()

    def exists(self, id: int) -> bool:
        return FuenteDatosParametros.objects.filter(pk=id, eliminado=False).exists()

    def list_all(self) -> list[FuenteDatosParametros]:
        return list(FuenteDatosParametros.objects.filter(eliminado=False))

    def save(self, entity: FuenteDatosParametros) -> None:
        entity.save()

    def update(self, id: int, entity: FuenteDatosParametros) -> None:
        FuenteDatosParametros.objects.filter(pk=id, eliminado=False).update(
            clave=entity.clave,
            valor=entity.valor,
            tipo_dato=entity.tipo_dato,
        )

    def delete(self, id: int) -> None:
        FuenteDatosParametros.objects.filter(pk=id, eliminado=False).update(
            eliminado=True
        )

    def create(self, **kwargs) -> FuenteDatosParametros:
        kwargs.setdefault("eliminado", False)
        return FuenteDatosParametros.objects.create(**kwargs)

    def get_by_fuente(self, fuente_datos_id: int) -> dict[str, str]:
        return {
            parametro.clave: parametro.valor
            for parametro in FuenteDatosParametros.objects.filter(
                fuente_datos_id=fuente_datos_id,
                eliminado=False,
            )
        }

    def list_by_fuente(self, fuente_datos_id: int) -> list[FuenteDatosParametros]:
        return list(
            FuenteDatosParametros.objects.filter(
                fuente_datos_id=fuente_datos_id,
                eliminado=False,
            )
        )

    def set_param(
        self,
        fuente_datos_id: int,
        clave: str,
        valor: str,
        tipo_dato: str = "string",
    ) -> None:
        FuenteDatosParametros.objects.update_or_create(
            fuente_datos_id=fuente_datos_id,
            clave=clave,
            defaults={
                "valor": valor,
                "tipo_dato": tipo_dato,
                "eliminado": False,
            },
        )

    def delete_by_fuente(self, fuente_datos_id: int) -> None:
        FuenteDatosParametros.objects.filter(
            fuente_datos_id=fuente_datos_id,
            eliminado=False,
        ).update(eliminado=True)
