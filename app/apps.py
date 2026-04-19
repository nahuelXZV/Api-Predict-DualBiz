from django.apps import AppConfig as DjangoAppConfig


class AppConfig(DjangoAppConfig):
    name = "app"
    verbose_name = "Api-Predict-DualBiz"

    def ready(self):
        from app.domain.core.logging import setup_logging, logger
        from app.domain.core.config import settings

        setup_logging()
        logger.info("startup", env=settings.app_env)
