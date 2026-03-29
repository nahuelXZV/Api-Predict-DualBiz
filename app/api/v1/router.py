from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import training
from app.api.v1.endpoints import models
from app.api.v1.endpoints import predict

router = APIRouter(prefix="/api/v1")
router.include_router(health.router, tags=[""])
router.include_router(training.router, tags=["training"])
router.include_router(models.router, tags=["models"])
router.include_router(predict.router, tags=["predict"])