from contextlib import asynccontextmanager
from loguru import logger
from pathlib import Path

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import (
    APIKeyMiddleware,
    RequestLoggingMiddleware,
    setup_cors,
)
from app.core.registry import ModelRegistry
from app.tasks.inference import InferenceExecutor
from app.api import health, models, predict

settings = get_settings()
setup_logging(settings.logging)

loader = None
inference_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Args:
        app: FastAPI application instance.
    """
    global loader, inference_executor

    logger.info("Starting X-AnyLabeling Server...")
    logger.info("Configuration loaded from configs/server.yaml")

    config_dir = Path(__file__).parent.parent / "configs"
    loader = ModelRegistry(config_dir)

    try:
        loader.load_all_models()
    except Exception as e:
        logger.critical(f"Failed to load models: {e}")
        raise

    inference_executor = InferenceExecutor(
        loader=loader,
        max_workers=settings.concurrency.max_workers,
        max_queue_size=settings.concurrency.max_queue_size,
    )

    yield

    logger.info("Shutting down server...")
    inference_executor.shutdown()
    loader.unload_all_models()
    logger.info("Server stopped")


app = FastAPI(
    title="X-AnyLabeling Server",
    description="AI Model Inference Service for X-AnyLabeling",
    version="0.0.1",
    lifespan=lifespan,
)

setup_cors(app, settings.security.cors_origins)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    APIKeyMiddleware,
    enabled=settings.security.api_key_enabled,
    api_key=settings.security.api_key,
    header_name=settings.security.api_key_header,
)

app.include_router(health.router, tags=["Health"])
app.include_router(models.router, tags=["Models"])
app.include_router(predict.router, tags=["Prediction"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=False,
    )
