from django.urls import path
from app.presentation.web.views import HomeView, ModelsView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("models/", ModelsView.as_view(), name="models"),
]
