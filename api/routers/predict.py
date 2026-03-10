from fastapi import APIRouter, HTTPException
from api.schemas.request import TicketFeatures
from api.schemas.response import PredictionResponse, SinglePrediction
from api.services.models import (
    get_all_models, encode_input, predict_with_confidence, get_production_model
)

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict_all(features: TicketFeatures):
    """Run all loaded models and return predictions side by side."""
    models = get_all_models()
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    X = encode_input(features.model_dump())

    predictions = []
    for name, model in models.items():
        price, low, high = predict_with_confidence(model, X)
        predictions.append(SinglePrediction(
            model_name=name,
            predicted_price_gbp=price,
            confidence_low_gbp=low,
            confidence_high_gbp=high,
        ))

    best = get_production_model() or predictions[0].model_name

    return PredictionResponse(
        predictions=predictions,
        best_model=best,
        input_summary={
            "origin": features.Origin,
            "destination": features.Destination,
            "seat_class": features.Seat_Class,
            "days_before_travel": features.Days_Before_Travel,
            "distance_km": features.Distance_km,
        }
    )


@router.post("/{model_name}", response_model=SinglePrediction)
def predict_single(model_name: str, features: TicketFeatures):
    """Run a specific model by name."""
    models = get_all_models()
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )

    X = encode_input(features.model_dump())
    price, low, high = predict_with_confidence(models[model_name], X)

    return SinglePrediction(
        model_name=model_name,
        predicted_price_gbp=price,
        confidence_low_gbp=low,
        confidence_high_gbp=high,
    )
