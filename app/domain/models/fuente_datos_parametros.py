from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.fuente_datos import FuenteDatos


class FuenteDatosParametros(BaseModelABC):
    fuente_datos = models.ForeignKey(
        FuenteDatos,
        on_delete=models.CASCADE,
        related_name="parametros",
    )
    clave = models.CharField(max_length=100)
    valor = models.TextField()
    tipo_dato = models.CharField(max_length=20, default="string")

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[fuente_datos_parametros]"
