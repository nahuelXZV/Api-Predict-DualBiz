from django.db import models
from app.domain.abstractions.base_model_abc import BaseModelABC
from .tarea_programada import TareaProgramada


class EjecucionTareaProgramada(BaseModelABC):
    tarea_programada = models.ForeignKey(
        TareaProgramada,
        on_delete=models.PROTECT,
        related_name="ejecuciones",
    )
    disparado_por = models.CharField(max_length=20)
    estado = models.CharField(max_length=20, default="pendiente")
    iniciado_en = models.DateTimeField()
    finalizado_en = models.DateTimeField(blank=True, null=True)
    mensaje_error = models.TextField(blank=True, null=True)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[ejecucion_tarea_programada]"
