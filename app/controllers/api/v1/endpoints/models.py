from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema

from app.controllers.api.responses import success_response
from app.controllers.api.v1.endpoints.serializers import ModelMetadataSerializer
from app.application.services.model_manager_service import ModelManagerService


class ModelsView(APIView):
    @extend_schema(
        tags=["models"],
        summary="Listar modelos cargados en el registro",
        responses={200: ModelMetadataSerializer(many=True)},
    )
    def get(self, request):
        service = ModelManagerService()
        result = service.list_models()
        return success_response(data=result, message="Modelos listados exitosamente.")
