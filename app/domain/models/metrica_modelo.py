from django.db import models

from .version_modelo import VersionModelo


class MetricaModelo(models.Model):
    version_modelo = models.ForeignKey(
        VersionModelo,
        on_delete=models.CASCADE,
        related_name="metricas",
    )
    nombre_metrica = models.CharField(max_length=100)
    valor_metrica = models.FloatField()
    split = models.CharField(max_length=20, blank=True, null=True)
    calculado_en = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "ml_metrica_modelo"
