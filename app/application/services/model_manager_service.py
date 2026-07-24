from app.application.ml.model_manager import model_manager
from app.domain.ml.model_metadata import ModelMetadata


class ModelManagerService:
    def list_models(self) -> list[ModelMetadata]:
        return model_manager.list_models()
