from app.domain.dtos.predict_dto import PredictResponseDTO
from app.ml.model_manager import model_manager

class PredictService:

    def predict(self, model_name: str, hyperparams: dict) -> PredictResponseDTO:
        response = model_manager.predict(
            model_name=model_name,
            data = hyperparams
        )
        print(f"PredictService: response={response}")
        return PredictResponseDTO (
            model_name=model_name,
            predictions=response.to_dict(orient="records"),
            success=True
        )
        
    