from __future__ import annotations

from typing import Callable

from app.domain.abstractions.data_writer_abc import DataWriterABC

_BUILDERS: dict[str, Callable[[dict], DataWriterABC]] = {}


def register_datawriter(writer_type: str) -> Callable:
    def decorator(fn: Callable[[dict], DataWriterABC]) -> Callable:
        _BUILDERS[writer_type] = fn
        return fn

    return decorator
