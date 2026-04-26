from django.db import models
from app.domain.models.base_model_abc import BaseModelABC
from app.domain.models.lote_prediccion import LotePrediccion


class ResultadoPrediccion(BaseModelABC):
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
    complementos = models.CharField(max_length=500, blank=True, null=True)

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[resultado_prediccion]"
