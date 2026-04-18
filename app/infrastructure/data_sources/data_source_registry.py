from __future__ import annotations

from typing import Callable

from app.domain.abstractions.data_source_abc import DataSourceABC

_BUILDERS: dict[str, Callable[[dict], DataSourceABC]] = {}


def register_datasource(source_type: str) -> Callable:
    def decorator(fn: Callable[[dict], DataSourceABC]) -> Callable:
        _BUILDERS[source_type] = fn
        return fn

    return decorator
