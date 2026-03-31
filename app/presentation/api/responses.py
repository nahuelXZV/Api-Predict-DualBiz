import dataclasses
from rest_framework.response import Response
from rest_framework import status as http_status

from app.domain.core.config import tz_now


def _build_envelope(success: bool, message: str, data, errors: list[str], status: int) -> Response:
    return Response(
        {
            "success": success,
            "message": message,
            "data": data,
            "errors": errors,
            "timestamp": tz_now().isoformat(),
        },
        status=status,
    )


def _serialize(data):
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return dataclasses.asdict(data)
    if isinstance(data, list) and data and dataclasses.is_dataclass(data[0]):
        return [dataclasses.asdict(item) for item in data]
    return data


def success_response(data=None, message: str = "OK", status: int = http_status.HTTP_200_OK) -> Response:
    return _build_envelope(True, message, _serialize(data), [], status)


def error_response(errors: list[str], message: str = "Error en la solicitud.", status: int = http_status.HTTP_400_BAD_REQUEST) -> Response:
    return _build_envelope(False, message, None, errors, status)
