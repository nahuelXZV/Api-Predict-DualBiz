from django.db import models
from app.domain.models.base_model_abc import BaseModelABC


class TareaParametro(BaseModelABC):
    tarea_programada = models.ForeignKey(
        "TareaProgramada",
        on_delete=models.CASCADE,
        related_name="parametros",
    )
    clave = models.CharField(max_length=100)
    valor = models.CharField(max_length=500)
    tipo_dato = models.CharField(max_length=20, default="string")

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[tarea_parametro]"
        unique_together = ("tarea_programada", "clave")
