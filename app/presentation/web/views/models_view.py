from django.shortcuts import render
from django.views import View

from app.application.services.model_manager_service import ModelManagerService


class ModelsView(View):
    def get(self, request):
        service = ModelManagerService()
        models = service.list_models()
        return render(request, "app/models.html", {"models": models})
