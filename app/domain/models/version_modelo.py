from django.db import models
from app.domain.abstractions.base_model_abc import BaseModelABC
from .ejecucion_tarea_programada import EjecucionTareaProgramada


class VersionModelo(BaseModelABC):
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
    parametros = models.JSONField(default=dict)
    activo = models.BooleanField(default=False)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[version_modelo]"
