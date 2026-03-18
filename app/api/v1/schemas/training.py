from pydantic import BaseModel

class TrainRequest(BaseModel):
    model_name:  str  = "knn"
    version:     str  = "1.0"
    hyperparams: dict = {}

class TrainResponse(BaseModel):
    model_name:     str
    version:        str
    steps_executed: list[str]
    errors:         list[str]
    success:        bool