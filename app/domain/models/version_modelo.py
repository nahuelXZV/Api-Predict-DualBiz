from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.ejecucion_tarea_programada import EjecucionTareaProgramada
from app.domain.models.fuente_datos import FuenteDatos


class VersionModelo(BaseModelABC):
    ejecucion_tarea_programada = models.OneToOneField(
        EjecucionTareaProgramada,
        on_delete=models.PROTECT,
        related_name="version_modelo",
        blank=True,
        null=True,
    )
    fuente_datos = models.ForeignKey(
        FuenteDatos,
        on_delete=models.PROTECT,
        related_name="fuente_datos_version",
    )
    nombre_modelo = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    entrenado_en = models.DateTimeField()
    ruta_pkl = models.CharField(max_length=500)
    parametros = models.JSONField(default=dict)
    activo = models.BooleanField(default=False)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[version_modelo]"
