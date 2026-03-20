from fastapi import APIRouter
from app.application.services.model_manager_service import ModelManagerService
from app.domain.dtos.model_metadata_dto import ModelMetadataDTO
from app.domain.dtos.response_dto import ResponseDTO

router = APIRouter()

@router.get("/list_models", response_model=ResponseDTO[list[ModelMetadataDTO]])
async def list_models():
    service = ModelManagerService()
    result  = service.list_models()
    return ResponseDTO(success=True, message="Models listed successfully", data=result)