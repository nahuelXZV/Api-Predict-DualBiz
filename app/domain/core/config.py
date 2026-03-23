from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ml-api"
    app_version: str = "1.0.0"   # ← agregar
    app_env: str = "development"
    app_debug: bool = False       # ← renombrar de debug a app_debug
    log_level: str = "INFO"

    path_models: str = "storage/models"
    path_data: str = "storage/data"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()