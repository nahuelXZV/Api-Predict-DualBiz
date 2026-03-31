from django.urls import path
from app.controllers.api.v1.endpoints.health import HealthView
from app.controllers.api.v1.endpoints.predict import PredictView
from app.controllers.api.v1.endpoints.training import TrainingView
from app.controllers.api.v1.endpoints.models import ModelsView

urlpatterns = [
    path("", HealthView.as_view(), name="health"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("train/", TrainingView.as_view(), name="train"),
    path("list_models/", ModelsView.as_view(), name="list_models"),
]
