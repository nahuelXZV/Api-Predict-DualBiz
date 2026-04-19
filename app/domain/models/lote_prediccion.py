from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.version_modelo import VersionModelo


class LotePrediccion(BaseModelABC):
    nombre_modelo = models.CharField(max_length=100, unique=True)
    version_modelo = models.ForeignKey(
        VersionModelo,
        on_delete=models.PROTECT,
        related_name="lote_prediccion_activo",
    )
    generado_en = models.DateTimeField()
    cantidad_predicciones = models.IntegerField(default=0)
    parametros = models.JSONField(default=dict)
    estado = models.CharField(max_length=20, default="generando")

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[lote_prediccion]"
