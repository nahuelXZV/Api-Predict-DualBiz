from django.db import models

from .lote_prediccion import LotePrediccion


class ResultadoPrediccion(models.Model):
    id = models.AutoField(primary_key=True)
    lote_prediccion = models.ForeignKey(
        LotePrediccion,
        on_delete=models.CASCADE,
        related_name="resultados",
    )
    cliente_id = models.CharField(max_length=50)
    producto_id = models.CharField(max_length=50)
    fuente = models.CharField(max_length=20)
    cantidad_sugerida = models.FloatField()
    score = models.FloatField()
    posicion = models.PositiveSmallIntegerField()

    class Meta:
        db_table = '[ml].[resultado_prediccion]'
