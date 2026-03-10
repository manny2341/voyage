from typing import Optional, Dict, List
from pydantic import BaseModel


class SinglePrediction(BaseModel):
    model_name: str
    predicted_price_gbp: float
    confidence_low_gbp: float
    confidence_high_gbp: float


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    best_model: str
    input_summary: Dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    total_models: int


class ModelsResponse(BaseModel):
    available_models: List[str]
    production_model: Optional[str]
