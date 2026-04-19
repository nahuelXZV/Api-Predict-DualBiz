from app.domain.dtos.training_dto import TrainRequestDTO, TrainResponseDTO
from app.application.ml.model_manager import model_manager


class TrainingService:
    def run(self, request: TrainRequestDTO) -> TrainResponseDTO:
        return model_manager.train(request)
