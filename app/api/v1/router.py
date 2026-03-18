from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import training

router = APIRouter(prefix="/api/v1")
router.include_router(health.router, tags=[""])
router.include_router(training.router, tags=["training"])