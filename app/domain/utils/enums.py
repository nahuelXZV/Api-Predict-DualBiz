from enum import Enum


class TipoJob(str, Enum):
    TRAINING = "training"
    PREDICT = "predict"


class EstadoEjecucion(str, Enum):
    PENDIENTE = "pendiente"
    EJECUTANDO = "ejecutando"
    EXITOSO = "exitoso"
    FALLIDO = "fallido"
    PENDIENTE_REINTENTO = "pendiente_reintento"


class DisparadoPor(str, Enum):
    SCHEDULER = "scheduler"
    MANUAL = "manual"
    REINTENTO = "reintento"
