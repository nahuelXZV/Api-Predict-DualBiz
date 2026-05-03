from app.domain.core.config import tz_now


def ObtenerVersion():
    return tz_now().strftime("%Y.%m.%d")
