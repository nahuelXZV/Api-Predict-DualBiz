from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ml-api"
    app_version: str = "1.0.0"
    app_env: str = "development"
    app_debug: bool = False
    log_level: str = "INFO"
    timezone: str = "America/La_Paz"

    path_models: str = "storage/models"
    path_data: str = "storage/data"

    ml_db_driver: str = "ODBC Driver 17 for SQL Server"
    ml_db_server: str = ""
    ml_db_database: str = ""
    ml_db_user: str = ""
    ml_db_password: str = ""

    @property
    def ml_db_connection_string(self) -> str:
        return (
            f"DRIVER={{{self.ml_db_driver}}};"
            f"SERVER={self.ml_db_server};"
            f"DATABASE={self.ml_db_database};"
            f"UID={self.ml_db_user};"
            f"PWD={self.ml_db_password}"
        )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()


def tz_now() -> datetime:
    """Retorna el datetime actual en la zona horaria configurada (default: America/La_Paz)."""
    return datetime.now(ZoneInfo(settings.timezone))
