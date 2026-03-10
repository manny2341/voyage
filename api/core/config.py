import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ENCODERS_PATH        = Path(os.getenv("ENCODERS_PATH", "data/processed/encoders.json"))
ACTIVE_MODEL_PATH    = Path(os.getenv("ACTIVE_MODEL_PATH", "data/processed/active_model.json"))

MODEL_NAMES = [
    "linear_regression",
    "ridge_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

CATEGORICAL_COLS = [
    "Seat_Class", "Booking_Channel", "Origin", "Destination",
    "Route_Category", "Customer_Segment", "Loyalty_Status",
]

NUMERIC_FEATURE_ORDER = [
    "Seat_Class", "Booking_Channel", "Origin", "Destination",
    "Distance_km", "Route_Category", "Customer_Segment", "Loyalty_Status",
    "Booking_Frequency_Qtr", "Average_Spend_GBP", "Total_Seats",
    "Seats_Sold_Realized", "Remaining_Seats_Realized", "Demand_Index",
    "Base_Price_At_Booking", "Days_Before_Travel", "Price_Premium",
    "Load_Factor", "Occupancy_Rate", "Price_Per_km",
    "Booking_Urgency", "Revenue_Potential",
]
