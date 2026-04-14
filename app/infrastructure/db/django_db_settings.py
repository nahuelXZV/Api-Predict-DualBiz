from app.domain.core.config import settings


def get_databases() -> dict:
    """
    Construye el dict DATABASES de Django según las variables de entorno.

    Si app_db_server está configurado usa SQL Server (mssql-django).
    Si no, cae a SQLite para desarrollo local.
    """
    if settings.app_db_server:
        return {
            "default": {
                "ENGINE": "mssql",
                "NAME": settings.app_db_database,
                "USER": settings.app_db_user,
                "PASSWORD": settings.app_db_password,
                "HOST": settings.app_db_server,
                "PORT": "",
                "OPTIONS": {
                    "driver": settings.app_db_driver,
                },
            }
        }

    # Fallback: SQLite para entorno local / CI sin SQL Server
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    return {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": base_dir / "db.sqlite3",
        }
    }
