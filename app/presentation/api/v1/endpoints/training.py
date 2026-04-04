from typing import cast

from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema

from app.domain.dtos.training_dto import TrainRequestDTO
from app.presentation.api.responses import success_response
from app.presentation.api.v1.endpoints.serializers import TrainRequestSerializer, TrainResponseSerializer
from app.application.services.training_service import TrainingService


class TrainingView(APIView):
    @extend_schema(
        tags=["training"],
        summary="Entrenar un modelo ML",
        request=TrainRequestSerializer,
        responses={200: TrainResponseSerializer},
    )
    def post(self, request):
        serializer = TrainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict, serializer.validated_data)

        service = TrainingService()
        result = service.run(TrainRequestDTO(
            model_name=data["model_name"],
            version=data["version"],
            data_source_config=data["data_source"],
        ))

        return success_response(data=result, message="Entrenamiento completado exitosamente.")
