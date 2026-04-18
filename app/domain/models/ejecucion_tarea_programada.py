from django.db import models

from .tarea_programada import TareaProgramada


class EjecucionTareaProgramada(models.Model):
    id = models.AutoField(primary_key=True)
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

    class Meta:
        db_table = '[ml].[ejecucion_tarea_programada]'
