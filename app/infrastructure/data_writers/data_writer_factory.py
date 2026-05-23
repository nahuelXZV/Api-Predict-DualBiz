from app.domain.abstractions.data_writer_abc import DataWriterABC
from app.infrastructure.data_writers.data_writer_registry import _BUILDERS


class DataWriterFactory:
    @staticmethod
    def build(writer_type: str, parameters: dict) -> DataWriterABC:
        if not isinstance(writer_type, str) or not writer_type:
            available = list(_BUILDERS.keys())
            raise ValueError(
                f"Tipo de datawriter invalido: '{writer_type}'. Opciones: {available}"
            )

        builder = _BUILDERS.get(writer_type)
        if builder is None:
            available = list(_BUILDERS.keys())
            raise ValueError(
                f"Tipo de datawriter desconocido: '{writer_type}'. Opciones: {available}"
            )
        return builder(parameters or {})
