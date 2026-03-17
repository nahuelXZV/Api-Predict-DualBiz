# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("startup", env=settings.app_env)
    yield
    logger.info("shutdown")

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

#app.add_exception_handler(ModelNotFoundError, model_not_found_handler)
#app.add_exception_handler(ModelNotReadyError, model_not_ready_handler)
app.include_router(router)