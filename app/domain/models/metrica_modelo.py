from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.version_modelo import VersionModelo


class MetricaModelo(BaseModelABC):
    version_modelo = models.ForeignKey(
        VersionModelo,
        on_delete=models.CASCADE,
        related_name="metricas",
    )
    nombre_metrica = models.CharField(max_length=100)
    valor_metrica = models.FloatField()
    split = models.CharField(max_length=20, blank=True, null=True)
    calculado_en = models.DateTimeField(auto_now_add=True)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[metrica_modelo]"
