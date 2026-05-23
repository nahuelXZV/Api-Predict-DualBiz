import pandas as pd
import pyodbc

from app.domain.abstractions.data_source_abc import DataSourceABC
from app.domain.core.logging import logger
from app.infrastructure.data_sources.data_source_registry import (
    register_datasource,
)
from app.infrastructure.db.sqlserver_connection import build_sqlserver_connection_string


class SqlServerDataSourceStrategy(DataSourceABC):
    def __init__(self, connection_string: str, query: str) -> None:
        self._connection_string = connection_string
        self._query = query

    def load(self) -> pd.DataFrame:
        batch_size = 5000
        chunks = []

        with pyodbc.connect(self._connection_string) as conn:
            logger.info("sqlserver_datasource_conectando")
            cursor = conn.cursor()
            cursor.execute(self._query)

            columns = [col[0] for col in cursor.description]

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                chunk_rows = [tuple(row) for row in rows]
                chunks.append(pd.DataFrame.from_records(chunk_rows, columns=columns))

        df = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=columns)
        )

        logger.info("sqlserver_datasource_cargado", filas=len(df), columnas=df.shape[1])
        return df


@register_datasource("sqlserver")
def _build(params: dict) -> SqlServerDataSourceStrategy:
    query = params.get("query") or ""
    if not query:
        raise ValueError("El datasource 'sqlserver' requiere el parametro 'query'.")

    conn_str = build_sqlserver_connection_string(params or {})
    logger.info("sqlserver_datasource_conexion_construida", connection_string=conn_str)
    return SqlServerDataSourceStrategy(conn_str, query)
