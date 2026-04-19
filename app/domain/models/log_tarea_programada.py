from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.ejecucion_tarea_programada import EjecucionTareaProgramada


class LogTareaProgramada(BaseModelABC):
    ejecucion_tarea_programada = models.ForeignKey(
        EjecucionTareaProgramada,
        on_delete=models.CASCADE,
        related_name="logs_steps",
    )
    nombre_step = models.CharField(max_length=100)
    orden_step = models.PositiveSmallIntegerField()
    estado = models.CharField(max_length=20)
    duracion_segundos = models.FloatField(blank=True, null=True)
    mensaje_error = models.TextField(blank=True, null=True)
    ejecutado_en = models.DateTimeField()

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[log_tarea_programada]"
