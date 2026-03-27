from app.domain.dtos.predict_dto import PredictResponseDTO
from app.domain.ml.model_metadata import ModelMetadata
from app.ml.model_manager import model_manager
from app.ml.models.pedido_sugerido_model import PedidoSugeridoModel
from app.domain.ml.model_registry import model_registry

class PredictService:

    def predict(self, model_name: str, hyperparams: dict) -> PredictResponseDTO:
        # path_model = "C:\\xNahuel\\Proyectos\\xApiPredict\\storage\\models\\modelo_pedido_sugerido_1.0.pkl"
        path_model = "C:\\xDesarrollo\\MachineLearning\\Api-Predict-DualBiz\\storage\\models\\modelo_pedido_sugerido_1.0.pkl"
        meta_data = ModelMetadata(
            name=model_name,
            version="1.0",
            path_model=path_model,
        )
        model = PedidoSugeridoModel(metadata=meta_data)
        model.load(path_model)
        model_registry.register(
            name=model_name,
            model=model,
        )
        print(model_registry.list_models())

        response = model_manager.predict(
            model_name=model_name,
            data = hyperparams
        )
        if response is None:
            return PredictResponseDTO (
                model_name=model_name,
                predictions=[],
                success=False
            )
        
        print(f"PredictService: response={response}")
        return PredictResponseDTO (
            model_name=model_name,
            predictions=response.to_dict(orient="records"),
            success=True
        )
        
    