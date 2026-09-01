from django.db import models


class BaseModelABC(models.Model):
    id = models.BigAutoField(primary_key=True)
    creado_en = models.DateTimeField(auto_now_add=True)
    actualizado_en = models.DateTimeField(auto_now=True)
    eliminado = models.BooleanField(default=False)
    
    class Meta:
        abstract = True
