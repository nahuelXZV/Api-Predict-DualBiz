from fastapi import APIRouter
from app.api.v1.schemas.training import TrainResponse
from app.services.training_service import TrainingService

router = APIRouter()

@router.get("/train", response_model=TrainResponse)
async def train(
    model_name: str = "knn",
    version:    str = "1.0",
):
    service = TrainingService()
    result  = service.run(
        model_name  = model_name,
        version     = version,
        hyperparams = {},
    )
    return result