from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ml-api"
    app_env: str = "development"
    debug: bool = False
    model_path: str = "models"
    log_level: str = "INFO"

    path_models: str = "storage/models"
    path_data: str = "storage/data"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()