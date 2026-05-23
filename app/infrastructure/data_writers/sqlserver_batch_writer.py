from collections.abc import Sequence

import pandas as pd
import pyodbc

from app.domain.abstractions.data_writer_abc import DataWriterABC
from app.domain.core.logging import logger
from app.infrastructure.data_writers.data_writer_registry import register_datawriter
from app.infrastructure.db.sqlserver_connection import build_sqlserver_connection_string


class SqlServerBatchWriter(DataWriterABC):
    def __init__(
        self,
        connection_string: str,
        table: str,
        columns: Sequence[str] | None = None,
        batch_size: int = 1000,
    ) -> None:
        if not table:
            raise ValueError("SqlServerBatchWriter requiere una tabla destino.")
        if batch_size <= 0:
            raise ValueError("SqlServerBatchWriter requiere batch_size mayor a cero.")

        self._connection_string = connection_string
        self._table = table
        self._columns = list(columns) if columns else None
        self._batch_size = batch_size

    def insert_many(self, data: pd.DataFrame) -> int:
        if data.empty:
            logger.info("sqlserver_datawriter_sin_filas", tabla=self._table)
            return 0

        columns = self._columns or list(data.columns)
        if not columns:
            raise ValueError("SqlServerBatchWriter requiere al menos una columna.")

        missing_columns = [column for column in columns if column not in data.columns]
        if missing_columns:
            raise ValueError(
                "El DataFrame no contiene las columnas requeridas: "
                + ", ".join(missing_columns)
                + "."
            )

        insert_sql = self._build_insert_sql(columns)
        rows_data = data.loc[:, columns].astype(object)
        rows_data = rows_data.where(pd.notna(rows_data), None)
        total_rows = len(rows_data)

        logger.info(
            "sqlserver_datawriter_insertando",
            tabla=self._table,
            filas=total_rows,
            columnas=len(columns),
            batch_size=self._batch_size,
        )

        with pyodbc.connect(self._connection_string, autocommit=False) as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            try:
                for start in range(0, total_rows, self._batch_size):
                    batch = rows_data.iloc[start : start + self._batch_size]
                    cursor.executemany(
                        insert_sql,
                        list(batch.itertuples(index=False, name=None)),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(
                    "sqlserver_datawriter_error_insertando", tabla=self._table
                )
                raise

        logger.info("sqlserver_datawriter_insertado", tabla=self._table, filas=total_rows)
        return total_rows

    def _build_insert_sql(self, columns: Sequence[str]) -> str:
        quoted_table = _quote_multipart_identifier(self._table)
        quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        return f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES ({placeholders})"


def _quote_multipart_identifier(identifier: str) -> str:
    parts = [part.strip() for part in identifier.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Identificador SQL Server invalido: '{identifier}'.")
    return ".".join(_quote_identifier(part) for part in parts)


def _quote_identifier(identifier: str) -> str:
    value = str(identifier or "").strip()
    if not value:
        raise ValueError("Los identificadores SQL Server no pueden estar vacios.")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return "[" + value.replace("]", "]]") + "]"


@register_datawriter("sqlserver")
def _build(params: dict) -> SqlServerBatchWriter:
    table = str(params.get("table") or "").strip()
    if not table:
        raise ValueError("El datawriter 'sqlserver' requiere el parametro 'table'.")

    batch_size = int(params.get("batch_size") or 1000)
    columns = params.get("columns")
    if columns is not None and not isinstance(columns, list):
        raise ValueError("El parametro 'columns' debe ser una lista de nombres.")

    conn_str = build_sqlserver_connection_string(params or {})
    logger.info("sqlserver_datawriter_conexion_construida", connection_string=conn_str)
    return SqlServerBatchWriter(conn_str, table, columns=columns, batch_size=batch_size)
