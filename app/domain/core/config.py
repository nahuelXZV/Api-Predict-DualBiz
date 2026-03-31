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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


def tz_now() -> datetime:
    """Retorna el datetime actual en la zona horaria configurada (default: America/La_Paz)."""
    return datetime.now(ZoneInfo(settings.timezone))