"""
models.py — Loads all trained models from pkl files at startup.
Caches them in memory — never loaded per-request.
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_models: Dict[str, object] = {}
_encoders: Dict = {}
_production_model: Optional[str] = None

MODELS_DIR = Path("data/processed/models")


def load_all_models(mlflow_uri: str, encoders_path: Path, active_model_path: Path):
    global _production_model

    # Load encoders
    if encoders_path.exists():
        with open(encoders_path) as f:
            _encoders.update(json.load(f))
        log.info(f"Encoders loaded from {encoders_path}")

    # Load active model config
    production = None
    if active_model_path.exists():
        with open(active_model_path) as f:
            config = json.load(f)
        production = config.get("production_model")

    # Load each pkl file
    if MODELS_DIR.exists():
        for pkl_file in sorted(MODELS_DIR.glob("*.pkl")):
            name = pkl_file.stem
            try:
                with open(pkl_file, "rb") as f:
                    model = pickle.load(f)
                _models[name] = model
                log.info(f"Loaded model: {name}")
            except Exception as e:
                log.warning(f"Could not load {name}: {e}")
    else:
        log.warning(f"Models directory not found: {MODELS_DIR}")

    _production_model = production or (list(_models.keys())[0] if _models else None)
    log.info(f"Models ready: {list(_models.keys())} | Production: {_production_model}")


def get_model(name: str):
    if name not in _models:
        raise KeyError(f"Model '{name}' not found. Available: {list(_models.keys())}")
    return _models[name]


def get_all_models() -> Dict:
    return _models


def get_encoders() -> Dict:
    return _encoders


def get_production_model() -> Optional[str]:
    return _production_model


def list_loaded_models():
    return list(_models.keys())


def encode_input(raw: dict) -> np.ndarray:
    from api.core.config import CATEGORICAL_COLS, NUMERIC_FEATURE_ORDER

    data = dict(raw)
    encoders = get_encoders()

    for col in CATEGORICAL_COLS:
        val = str(data.get(col, "Unknown"))
        mapping = encoders.get(col, {})
        data[col] = mapping.get(val, 0)

    total    = data.get("Total_Seats", 1) or 1
    sold     = data.get("Seats_Sold_Realized", 0)
    distance = data.get("Distance_km", 1) or 1
    base     = data.get("Base_Price_At_Booking", 1)
    days     = data.get("Days_Before_Travel", 1)

    data["Occupancy_Rate"]    = sold / total
    data["Price_Per_km"]      = base / distance
    data["Booking_Urgency"]   = round(1 / (days + 1), 6)
    data["Revenue_Potential"] = base * total

    vector = [data.get(col, 0) for col in NUMERIC_FEATURE_ORDER]
    return np.array(vector).reshape(1, -1)


def predict_with_confidence(model, X: np.ndarray) -> tuple:
    pred = float(model.predict(X)[0])
    try:
        preds = np.array([est.predict(X)[0] for est in model.estimators_])
        std   = preds.std()
        low   = max(0, round(pred - 1.5 * std, 2))
        high  = round(pred + 1.5 * std, 2)
    except AttributeError:
        margin = pred * 0.12
        low    = max(0, round(pred - margin, 2))
        high   = round(pred + margin, 2)
    return round(pred, 2), low, high
