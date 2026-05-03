import pandas as pd
import pyodbc

from app.domain.abstractions.data_source_abc import DataSourceABC
from app.domain.core.logging import logger
from app.infrastructure.data_sources.data_source_registry import (
    register_datasource,
)


class SqlServerDataSourceStrategy(DataSourceABC):
    """
    Obtiene datos desde SQL Server ejecutando una query cruda con pyodbc.
    Usa pd.read_sql() directamente sobre la conexion, sin ORM, para
    maximizar velocidad en volumenes grandes.

    Args:
        connection_string: Cadena de conexion pyodbc.
        query: Query SQL a ejecutar.
    """

    def __init__(self, connection_string: str, query: str) -> None:
        self._connection_string = connection_string
        self._query = query

    def load(self) -> pd.DataFrame:
        logger.info("sqlserver_datasource_conectando")
        with pyodbc.connect(self._connection_string) as conn:
            df = pd.read_sql(self._query, conn)
        logger.info("sqlserver_datasource_cargado", filas=len(df), columnas=df.shape[1])
        return df


def _build_connection_string(params: dict) -> str:
    driver = str(params.get("driver") or "").strip()
    server = str(params.get("server") or "").strip()
    database = str(params.get("database") or "").strip()
    user = str(params.get("user") or params.get("uid") or "").strip()
    password = str(params.get("password") or params.get("pwd") or "").strip()
    port = str(params.get("port") or "").strip()
    trusted_connection = str(params.get("trusted_connection") or "").strip().lower()
    encrypt = str(params.get("encrypt") or "").strip()
    trust_server_certificate = str(params.get("trust_server_certificate") or "").strip()

    missing = [
        name
        for name, value in (
            ("driver", driver),
            ("server", server),
            ("database", database),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "El datasource 'sqlserver' requiere los parametros: "
            + ", ".join(missing)
            + "."
        )

    if port:
        server = f"{server},{port}"

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
    ]

    if trusted_connection in {"true", "1", "yes", "y", "sspi"}:
        parts.append("Trusted_Connection=yes")
    else:
        if not user or not password:
            raise ValueError(
                "El datasource 'sqlserver' requiere 'user' y 'password', "
                "o bien 'trusted_connection=true'."
            )
        parts.extend([f"UID={user}", f"PWD={password}"])

    if encrypt:
        parts.append(f"Encrypt={encrypt}")
    if trust_server_certificate:
        parts.append(f"TrustServerCertificate={trust_server_certificate}")

    return ";".join(parts)


@register_datasource("sqlserver")
def _build(params: dict) -> SqlServerDataSourceStrategy:
    query = params.get("query") or ""
    if not query:
        raise ValueError("El datasource 'sqlserver' requiere el parametro 'query'.")

    conn_str = _build_connection_string(params or {})
    return SqlServerDataSourceStrategy(conn_str, query)
