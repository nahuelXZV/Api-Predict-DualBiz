import json

from fastapi import APIRouter, Query
from app.application.services.predict_service import PredictService
from app.domain.dtos.predict_dto import PredictResponseDTO
from app.application.services.training_service import TrainingService
from app.domain.dtos.response_dto import ResponseDTO

router = APIRouter()

@router.get("/predict", response_model=ResponseDTO[PredictResponseDTO])
async def predict( 
    model_name: str = Query("knn", description="Nombre del modelo"),
    hyperparams: str = Query(
        default="{}",
        description="Hiperparámetros en formato JSON. Ejemplo: {\"learning_rate\": 0.05, \"max_depth\": 15}"
    )):
    service = PredictService()
    params_dict = json.loads(hyperparams)
    result  = service.predict(
        model_name  = model_name,
        hyperparams = params_dict,
    )
    
    return ResponseDTO(success=True, message="Prediction completed successfully", data=result)