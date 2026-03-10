"""
prepare_data.py — Reproducible preprocessing pipeline
Reads data/raw/tickets.csv → writes data/processed/tickets_processed.csv
Also saves encoder mappings to data/processed/encoders.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH       = Path("data/raw/tickets.csv")
PROCESSED_DIR  = Path("data/processed")
PROCESSED_PATH = PROCESSED_DIR / "tickets_processed.csv"
ENCODERS_PATH  = PROCESSED_DIR / "encoders.json"

CATEGORICAL_COLS = [
    "Seat_Class", "Booking_Channel", "Origin", "Destination",
    "Route_Category", "Customer_Segment", "Loyalty_Status",
]

NUMERIC_COLS = [
    "Distance_km", "Booking_Frequency_Qtr", "Average_Spend_GBP",
    "Total_Seats", "Seats_Sold_Realized", "Remaining_Seats_Realized",
    "Demand_Index", "Base_Price_At_Booking", "Days_Before_Travel",
    "Price_Premium", "Load_Factor",
]

TARGET = "Ticket_Price_GBP"


def validate(df: pd.DataFrame):
    """Basic schema and quality checks."""
    required = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    null_counts = df[required].isnull().sum()
    if null_counts.any():
        print(f"Warning — nulls found:\n{null_counts[null_counts > 0]}")

    assert df[TARGET].min() >= 0, "Negative ticket prices found"
    assert df["Distance_km"].min() >= 0, "Negative distances found"
    assert df["Load_Factor"].between(0, 1).all(), "Load factor out of [0,1] range"
    print("Validation passed.")


def encode_categoricals(df: pd.DataFrame):
    """Ordinal-encode all categorical columns. Save mappings."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
        cats = sorted(df[col].astype(str).unique().tolist())
        mapping = {v: i for i, v in enumerate(cats)}
        df[col] = df[col].astype(str).map(mapping)
        encoders[col] = mapping
    return df, encoders


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features."""
    df["Occupancy_Rate"]       = df["Seats_Sold_Realized"] / df["Total_Seats"].replace(0, 1)
    df["Price_Per_km"]         = df["Base_Price_At_Booking"] / df["Distance_km"].replace(0, 1)
    df["Booking_Urgency"]      = (1 / (df["Days_Before_Travel"] + 1)).round(6)
    df["Revenue_Potential"]    = df["Base_Price_At_Booking"] * df["Total_Seats"]
    return df


def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"  {len(df)} rows loaded")

    validate(df)

    df = add_features(df)
    df, encoders = encode_categoricals(df)

    # Time-based split: sort by Days_Before_Travel as proxy for booking time
    df = df.sort_values("Days_Before_Travel").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    with open(ENCODERS_PATH, "w") as f:
        json.dump(encoders, f, indent=2)

    print(f"Processed data saved → {PROCESSED_PATH}")
    print(f"Encoders saved       → {ENCODERS_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c != TARGET]}")


if __name__ == "__main__":
    main()
