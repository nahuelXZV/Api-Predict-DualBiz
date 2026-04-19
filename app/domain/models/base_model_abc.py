from django.db import models


class BaseModelABC(models.Model):
    id = models.AutoField(primary_key=True)
    eliminado = models.BooleanField(default=False)

    class Meta:
        abstract = True
