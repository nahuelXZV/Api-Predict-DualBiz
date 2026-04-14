from django.db import models


class TareaProgramada(models.Model):
    id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100, unique=True)
    nombre_modelo = models.CharField(max_length=100)
    tipo_job = models.CharField(max_length=30)
    cron_schedule = models.CharField(max_length=50, blank=True, null=True)
    activo = models.BooleanField(default=True)
    configuracion = models.JSONField(default=dict)
    creado_en = models.DateTimeField(auto_now_add=True)
    actualizado_en = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ml_tarea_programada"
