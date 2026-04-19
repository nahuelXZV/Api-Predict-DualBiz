from app.domain.abstractions.data_source_abc import DataSourceABC
from app.infrastructure.data_sources.data_source_registry import _BUILDERS


class DataSourceFactory:
    @staticmethod
    def build(parameters: dict) -> DataSourceABC:
        source_type = parameters.get("data_source_type")
        if not isinstance(source_type, str) or not source_type:
            available = list(_BUILDERS.keys())
            raise ValueError(
                f"Tipo de datasource inválido: '{source_type}'. Opciones: {available}"
            )

        builder = _BUILDERS.get(source_type)
        if builder is None:
            available = list(_BUILDERS.keys())
            raise ValueError(
                f"Tipo de datasource desconocido: '{source_type}'. Opciones: {available}"
            )
        return builder(parameters or {})
