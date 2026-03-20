from app.domain.dtos.model_metadata_dto import ModelMetadataDTO
from app.ml.model_manager import model_manager

class ModelManagerService:

    def list_models(self) -> list[ModelMetadataDTO]:
        return model_manager.list_models()
        
    