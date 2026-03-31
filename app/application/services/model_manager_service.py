from app.domain.ml.model_metadata import ModelMetadata
from app.infrastructure.ml.model_manager import model_manager


class ModelManagerService:

    def list_models(self) -> list[ModelMetadata]:
        return model_manager.list_models()
        
    