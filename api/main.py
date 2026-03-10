"""
main.py — FastAPI application entry point
Port: 8000
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import MLFLOW_TRACKING_URI, ENCODERS_PATH, ACTIVE_MODEL_PATH
from api.services.models import load_all_models, list_loaded_models, get_production_model
from api.schemas.response import HealthResponse, ModelsResponse
from api.routers import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup, release on shutdown."""
    log.info("Loading models from MLflow...")
    try:
        load_all_models(MLFLOW_TRACKING_URI, ENCODERS_PATH, ACTIVE_MODEL_PATH)
        log.info(f"Models ready: {list_loaded_models()}")
    except Exception as e:
        log.error(f"Model loading failed: {e}")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Voyage Ticket Price API",
    description="Predict railway ticket prices using trained ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://ui:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    loaded = list_loaded_models()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        total_models=len(loaded),
    )


@app.get("/models", response_model=ModelsResponse, tags=["system"])
def models():
    return ModelsResponse(
        available_models=list_loaded_models(),
        production_model=get_production_model(),
    )


@app.get("/", tags=["system"])
def root():
    return {
        "service": "Voyage Ticket Price API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "predict_all": "POST /predict",
        "predict_single": "POST /predict/{model_name}",
    }
