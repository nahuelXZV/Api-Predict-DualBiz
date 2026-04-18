from app.domain.dtos.predict_dto import PredictResponseDTO
from app.application.ml.model_manager import model_manager


class PredictService:
    def predict(self, model_name: str, hyperparams: dict) -> PredictResponseDTO:
        response = model_manager.predict(model_name=model_name, data=hyperparams)
        if response is None or "error" in response:
            return PredictResponseDTO(
                model_name=model_name, predictions=response or {}, success=False
            )

        return PredictResponseDTO(
            model_name=model_name, predictions=response, success=True
        )
