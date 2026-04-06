from app.domain.dtos.training_dto import TrainRequestDTO, TrainResponseDTO
from app.infrastructure.ml.model_manager import model_manager


class TrainingService:
    def run(self, request: TrainRequestDTO) -> TrainResponseDTO:
        return model_manager.train(request)
