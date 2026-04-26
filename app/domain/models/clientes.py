from django.db import models
from app.domain.models.base_model_abc import BaseModelABC

class Clientes(BaseModelABC):
    nombre_cliente = models.CharField(max_length=200)
    codigo_erp = models.CharField(max_length=100)
    registrado_en = models.DateTimeField()

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[clientes]"
