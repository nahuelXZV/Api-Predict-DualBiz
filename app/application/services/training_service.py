from app.domain.dtos.training_dto import TrainResponseDTO
from app.ml.model_manager import model_manager

class TrainingService:

    def run(self, model_name: str, version: str, hyperparams: dict) -> TrainResponseDTO:
        return model_manager.train(
            model_name = model_name,
            version    = version,
            hyperparams = hyperparams
        )
        
    