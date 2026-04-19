from enum import Enum


class TipoJob(str, Enum):
    TRAINING = "training"
    PREDICT = "predict"


class EstadoEjecucion(str, Enum):
    PENDIENTE = "pendiente"
    EJECUTANDO = "ejecutando"
    EXITOSO = "exitoso"
    FALLIDO = "fallido"


class DisparadoPor(str, Enum):
    SCHEDULER = "scheduler"
    MANUAL = "manual"
