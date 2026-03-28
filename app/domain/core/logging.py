import logging
import logging.handlers
import structlog
from pathlib import Path
from app.domain.core.config import settings, tz_now


def _tz_stamper(_logger, _method, event_dict):
    """Agrega el timestamp en la zona horaria configurada (settings.timezone)."""
    event_dict["timestamp"] = tz_now().isoformat()
    return event_dict


_SHARED_PROCESSORS = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    _tz_stamper,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
]


def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # --- Formatter para consola (legible en dev, JSON en prod) ---
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer() if settings.app_debug
            else structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    # --- Formatter para archivo (siempre JSON) ---
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    # --- Handler: consola ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # --- Handler: archivo rotativo diario (30 días de retención) ---
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.suffix = "%Y-%m-%d"  # app.log.2026-03-28

    # --- Root logger con ambos handlers ---
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # --- structlog enruta a través de stdlib ---
    structlog.configure(
        processors=_SHARED_PROCESSORS + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger()