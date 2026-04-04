import dataclasses as dc

from rest_framework.response import Response
from rest_framework import status as http_status

from app.domain.core.config import tz_now
from app.domain.dtos.response_dto import ResponseEnvelope


def _build_envelope(envelope: ResponseEnvelope) -> Response:
    return Response(
        {
            "success": envelope.success,
            "message": envelope.message,
            "data": envelope.data,
            "errors": envelope.errors,
            "timestamp": tz_now().isoformat(),
        },
        status=envelope.status,
    )


def _serialize(data):
    if dc.is_dataclass(data) and not isinstance(data, type):
        return dc.asdict(data)
    if isinstance(data, list) and data and dc.is_dataclass(data[0]):
        return [dc.asdict(item) for item in data]
    return data


def success_response(
    data=None, message: str = "OK", status: int = http_status.HTTP_200_OK
) -> Response:
    return _build_envelope(
        ResponseEnvelope(True, message, _serialize(data), [], status)
    )


def error_response(
    errors: list[str],
    message: str = "Error en la solicitud.",
    status: int = http_status.HTTP_400_BAD_REQUEST,
) -> Response:
    return _build_envelope(ResponseEnvelope(False, message, None, errors, status))
