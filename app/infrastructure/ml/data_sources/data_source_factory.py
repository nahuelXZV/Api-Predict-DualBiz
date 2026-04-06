from app.domain.core.config import settings
from app.domain.ml.abstractions.data_source_abc import DataSourceABC
from app.infrastructure.ml.data_sources.csv_data_source_strategy import (
    CsvDataSourceStrategy,
)
from app.infrastructure.ml.data_sources.sqlserver_data_source_strategy import (
    SqlServerDataSourceStrategy,
)


class DataSourceFactory:
    @staticmethod
    def build(config: dict) -> DataSourceABC:
        source_type = config.get("type")
        params: dict = config.get("params") or {}

        if source_type == "sqlserver":
            conn_str = (
                params.get("connection_string") or settings.ml_db_connection_string
            )
            query = params.get("query") or ""
            if not query:
                raise ValueError(
                    "El datasource 'sqlserver' requiere el parámetro 'params.query'."
                )
            return SqlServerDataSourceStrategy(conn_str, query)

        if source_type == "csv":
            path = params.get("path") or ""
            if not path:
                raise ValueError("El datasource 'csv' requiere el parámetro 'path'.")
            return CsvDataSourceStrategy(
                path,
                separator=params.get("separator", ","),
                encoding=params.get("encoding", "utf-8"),
            )

        raise ValueError(
            f"Tipo de datasource desconocido: '{source_type}'. Opciones: sqlserver, csv."
        )
