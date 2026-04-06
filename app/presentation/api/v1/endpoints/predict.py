from typing import cast

from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema

from app.presentation.api.responses import success_response
from app.presentation.api.v1.endpoints.serializers import (
    PredictRequestSerializer,
    PredictResponseSerializer,
)
from app.application.services.predict_service import PredictService


class PredictView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = PredictService()

    @extend_schema(
        tags=["predict"],
        summary="Realizar predicción con un modelo cargado",
        request=PredictRequestSerializer,
        responses={200: PredictResponseSerializer},
    )
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict, serializer.validated_data)

        result = self.service.predict(
            model_name=data["model_name"],
            hyperparams=data["parameters"],
        )

        return success_response(
            data=result, message="Predicción completada exitosamente."
        )
