"""
test_api.py — Basic API tests using FastAPI TestClient
Run: pytest tests/
"""

import pytest
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "Seat_Class": "Standard",
    "Booking_Channel": "Web",
    "Origin": "London",
    "Destination": "Paris",
    "Distance_km": 340.0,
    "Route_Category": "Medium",
    "Customer_Segment": "Leisure",
    "Loyalty_Status": "Gold",
    "Booking_Frequency_Qtr": 4,
    "Average_Spend_GBP": 150.0,
    "Total_Seats": 200,
    "Seats_Sold_Realized": 150,
    "Remaining_Seats_Realized": 50,
    "Demand_Index": 1.5,
    "Base_Price_At_Booking": 85.0,
    "Days_Before_Travel": 14,
    "Price_Premium": 1.15,
    "Load_Factor": 0.75,
}


def get_client():
    """Import here so models don't need to be loaded in CI."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from api.main import app
    return TestClient(app)


def test_health():
    client = get_client()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "models_loaded" in data


def test_models_endpoint():
    client = get_client()
    r = client.get("/models")
    assert r.status_code == 200
    assert "available_models" in r.json()


def test_predict_all_valid():
    client = get_client()
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert "best_model" in data
    for pred in data["predictions"]:
        assert pred["predicted_price_gbp"] > 0
        assert pred["confidence_low_gbp"] <= pred["predicted_price_gbp"]
        assert pred["confidence_high_gbp"] >= pred["predicted_price_gbp"]


def test_predict_invalid_seat_class():
    client = get_client()
    bad_payload = {**VALID_PAYLOAD, "Seat_Class": "Economy"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_single_model():
    client = get_client()
    r = client.post("/predict/random_forest", json=VALID_PAYLOAD)
    # 200 if model loaded, 404 if not
    assert r.status_code in (200, 404, 503)


def test_predict_unknown_model():
    client = get_client()
    r = client.post("/predict/unknown_model", json=VALID_PAYLOAD)
    assert r.status_code in (404, 503)
