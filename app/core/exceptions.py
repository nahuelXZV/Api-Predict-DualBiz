from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class ModelNotFoundError(Exception):
    def __init__(self, name: str):
        self.name = name

class ModelNotReadyError(Exception):
    pass

# Handlers que se registran en main.py
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=404, content={"detail": f"Modelo '{exc.name}' no encontrado"})

async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"detail": "Modelo no está listo"})