from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema


class HealthView(APIView):
    @extend_schema(
        tags=["health"],
        summary="Health check",
        responses={200: {"type": "object", "properties": {"status": {"type": "string"}}}},
    )
    def get(self, request):
        return Response({"status": "ok"})
