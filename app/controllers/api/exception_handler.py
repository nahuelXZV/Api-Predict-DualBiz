from rest_framework.views import exception_handler
from rest_framework import status

from app.domain.core.logging import logger
from app.controllers.api.responses import error_response


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    view_name = context["view"].__class__.__name__

    if response is not None:
        logger.warning(
            "api_error",
            status=response.status_code,
            detail=response.data,
            view=view_name,
        )
        return error_response(
            errors=_extract_errors(response.data),
            message="Error en la solicitud.",
            status=response.status_code,
        )

    logger.error("api_unhandled_exception", exc=str(exc), view=view_name)
    return error_response(
        errors=[str(exc)],
        message="Error interno del servidor.",
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _extract_errors(data) -> list[str]:
    if isinstance(data, list):
        return [str(e) for e in data]
    if isinstance(data, dict):
        errors = []
        for key, val in data.items():
            if isinstance(val, list):
                errors.extend([f"{key}: {e}" for e in val])
            else:
                errors.append(f"{key}: {val}")
        return errors
    return [str(data)]
