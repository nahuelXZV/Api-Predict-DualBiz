from fastapi import APIRouter
from app.domain.dtos.training_dto import TrainResponseDTO
from app.application.services.training_service import TrainingService
from app.domain.dtos.response_dto import ResponseDTO

router = APIRouter()

@router.get("/train", response_model=ResponseDTO[TrainResponseDTO])
async def train(model_name: str = "pedido_sugerido", version: str = "1.0",):
    service = TrainingService()
    result  = service.run(
        model_name  = model_name,
        version     = version,
        hyperparams = {},
    )
    return ResponseDTO(success=True, message="Training completed successfully", data=result)