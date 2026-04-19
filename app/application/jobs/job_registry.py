from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.domain.utils.enums import TipoJob

if TYPE_CHECKING:
    from app.domain.models.tarea_programada import TareaProgramada

_HANDLERS: dict[TipoJob, Callable[[TareaProgramada], None]] = {}


def register_job(tipo: TipoJob) -> Callable:
    def decorator(fn: Callable[[TareaProgramada], None]) -> Callable:
        _HANDLERS[tipo] = fn
        return fn

    return decorator


def get_handler(tipo: TipoJob) -> Callable[[TareaProgramada], None]:
    handler = _HANDLERS.get(tipo)
    if handler is None:
        available = [t.value for t in _HANDLERS]
        raise ValueError(
            f"Tipo de job no registrado: '{tipo.value}'. Disponibles: {available}"
        )
    return handler
