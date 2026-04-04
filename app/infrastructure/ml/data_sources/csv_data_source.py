import pandas as pd

from app.domain.core.logging import logger
from app.domain.ml.data_source import DataSource


class CsvDataSource(DataSource):
    """
    Obtiene datos desde un archivo CSV.
    Útil para desarrollo local, testing, o cuando el cliente
    entrega datos históricos en planilla.

    Args:
        path: Ruta al archivo CSV.
        separator: Separador de columnas (default ',').
        encoding: Encoding del archivo (default 'utf-8').
    """

    def __init__(
        self, path: str, separator: str = ",", encoding: str = "utf-8"
    ) -> None:
        self._path = path
        self._separator = separator
        self._encoding = encoding

    def load(self) -> pd.DataFrame:
        logger.info(
            "csv_datasource_cargando", path=self._path, separator=self._separator
        )
        df = pd.read_csv(self._path, sep=self._separator, encoding=self._encoding)
        logger.info("csv_datasource_cargado", filas=len(df), columnas=df.shape[1])
        return df
