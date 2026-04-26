from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import models
from app.domain.models.base_model_abc import BaseModelABC

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from app.domain.models.tarea_parametro import TareaParametro


class TareaProgramada(BaseModelABC):
    parametros: QuerySet[TareaParametro]
    nombre = models.CharField(max_length=100, unique=True)
    tipo_job = models.CharField(max_length=30)
    cron_schedule = models.CharField(max_length=50, blank=True, null=True)
    activo = models.BooleanField(default=True)
    max_reintentos = models.PositiveSmallIntegerField(default=0)
    delay_reintento_segundos = models.PositiveIntegerField(default=300)
    creado_en = models.DateTimeField(auto_now_add=True)
    actualizado_en = models.DateTimeField(auto_now=True)

    def get_params(self) -> dict:
        return {p.clave: p.valor for p in self.parametros.all()}

    class Meta(BaseModelABC.Meta):
        db_table = "[ml].[tarea_programada]"
