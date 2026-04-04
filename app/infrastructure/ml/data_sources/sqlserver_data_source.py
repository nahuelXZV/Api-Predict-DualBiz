import pandas as pd

from app.domain.core.logging import logger
from app.domain.ml.data_source import DataSource


class SqlServerDataSource(DataSource):
    """
    Obtiene datos desde SQL Server ejecutando una query cruda con pyodbc.
    Usa pd.read_sql() directamente sobre la conexión — sin ORM — para
    maximizar velocidad en volúmenes grandes.

    Args:
        connection_string: Cadena de conexión pyodbc.
            Ejemplo: "DRIVER={ODBC Driver 17 for SQL Server};SERVER=...;DATABASE=...;UID=...;PWD=..."
        query: Query SQL a ejecutar. Puede ser un SELECT directo o llamada
            a una vista. Se define en el módulo de cada modelo (queries.py).
    """

    def __init__(self, connection_string: str, query: str) -> None:
        self._connection_string = connection_string
        self._query = query

    def load(self) -> pd.DataFrame:
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc no está instalado. Ejecutá: pip install pyodbc")

        logger.info("sqlserver_datasource_conectando")
        with pyodbc.connect(self._connection_string) as conn:
            df = pd.read_sql(self._query, conn)
        logger.info("sqlserver_datasource_cargado", filas=len(df), columnas=df.shape[1])
        return df
