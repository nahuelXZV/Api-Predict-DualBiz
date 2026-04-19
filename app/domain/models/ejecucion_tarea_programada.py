from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.tarea_programada import TareaProgramada


class EjecucionTareaProgramada(BaseModelABC):
    tarea_programada = models.ForeignKey(
        TareaProgramada,
        on_delete=models.PROTECT,
        related_name="ejecuciones",
    )
    ejecucion_original = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reintentos",
    )
    disparado_por = models.CharField(max_length=20)
    estado = models.CharField(max_length=20, default="pendiente")
    numero_intento = models.PositiveSmallIntegerField(default=1)
    iniciado_en = models.DateTimeField()
    finalizado_en = models.DateTimeField(blank=True, null=True)
    mensaje_error = models.TextField(blank=True, null=True)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[ejecucion_tarea_programada]"
