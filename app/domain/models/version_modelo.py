from django.db import models

from .ejecucion_tarea_programada import EjecucionTareaProgramada


class VersionModelo(models.Model):
    id = models.AutoField(primary_key=True)
    ejecucion_tarea_programada = models.OneToOneField(
        EjecucionTareaProgramada,
        on_delete=models.PROTECT,
        related_name="version_modelo",
        blank=True,
        null=True,
    )
    nombre_modelo = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    entrenado_en = models.DateTimeField()
    ruta_pkl = models.CharField(max_length=500)
    tipo_fuente_datos = models.CharField(max_length=20)
    cantidad_clientes = models.IntegerField(default=0)
    cantidad_productos = models.IntegerField(default=0)
    hiperparametros = models.JSONField(default=dict)
    activo = models.BooleanField(default=False)

    class Meta:
        db_table = "ml_version_modelo"
