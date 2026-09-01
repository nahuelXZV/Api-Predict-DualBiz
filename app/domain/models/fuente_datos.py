from django.db import models

from app.domain.abstractions.base_model_abc import BaseModelABC


class FuenteDatos(BaseModelABC):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    tipo = models.CharField(max_length=50)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[fuente_datos]"
