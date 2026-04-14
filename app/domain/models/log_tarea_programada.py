from django.db import models

from .ejecucion_tarea_programada import EjecucionTareaProgramada


class LogTareaProgramada(models.Model):
    id = models.AutoField(primary_key=True)
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

    class Meta:
        db_table = "ml_log_tarea_programada"
