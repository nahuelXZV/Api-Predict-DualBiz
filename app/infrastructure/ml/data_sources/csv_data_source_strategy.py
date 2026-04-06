import pandas as pd

from app.domain.core.config import settings
from app.domain.core.logging import logger
from app.domain.ml.abstractions.data_source_abc import DataSourceABC


class CsvDataSourceStrategy(DataSourceABC):
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
        path_completo = settings.path_data + self._path
        df = pd.read_csv(path_completo, sep=self._separator, encoding=self._encoding)
        logger.info("csv_datasource_cargado", filas=len(df), columnas=df.shape[1])
        return df
