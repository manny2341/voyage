"""
streamlit_app.py — Voyage standalone Streamlit app for Streamlit Cloud.
Loads models directly from pkl files — no FastAPI layer needed.
"""

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR         = Path("data/processed/models")
ENCODERS_PATH      = Path("data/processed/encoders.json")
ACTIVE_MODEL_PATH  = Path("data/processed/active_model.json")

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

# ── Load models once at startup ───────────────────────────────────────────────

@st.cache_resource
def load_assets():
    with open(ENCODERS_PATH) as f:
        encoders = json.load(f)

    with open(ACTIVE_MODEL_PATH) as f:
        active = json.load(f)

    production_model = active.get("production_model")
    models = {}
    for pkl_file in sorted(MODELS_DIR.glob("*.pkl")):
        if pkl_file.stem != production_model:
            continue
        with open(pkl_file, "rb") as f:
            models[pkl_file.stem] = pickle.load(f)

    return models, encoders, production_model


models, encoders, production_model = load_assets()

# ── Encode input ──────────────────────────────────────────────────────────────

def encode_input(raw: dict) -> np.ndarray:
    data = dict(raw)
    for col in CATEGORICAL_COLS:
        val     = str(data.get(col, "Unknown"))
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

    return np.array([data.get(col, 0) for col in NUMERIC_FEATURE_ORDER]).reshape(1, -1)


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

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Voyage — Ticket Price Forecasting",
    page_icon="🚄",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .price-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 20px; text-align: center;
        margin-bottom: 12px;
    }
    .price-big { font-size: 2.4rem; font-weight: 800; color: #f1f5f9; }
    .price-range { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .best-badge {
        background: rgba(59,130,246,0.15); border: 1px solid #3b82f6;
        color: #60a5fa; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 700; display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚄 Voyage — Ticket Price Forecasting")
st.markdown(f"Enter journey details to get AI-powered price predictions across all models. "
            f"**{len(models)} models loaded | Production: {production_model}**")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🗺️ Route Details")
        origin      = st.selectbox("Origin",      ["London","Manchester","Birmingham","Edinburgh","Bristol","Leeds","Glasgow"])
        destination = st.selectbox("Destination", ["Paris","Amsterdam","Brussels","Frankfurt","Lyon","Cologne","Rotterdam"])
        distance_km = st.number_input("Distance (km)", min_value=50.0, max_value=2000.0, value=340.0, step=10.0)
        route_cat   = st.selectbox("Route Category", ["Short", "Medium", "Long"])

    with col2:
        st.markdown("#### 💺 Seat & Capacity")
        seat_class  = st.selectbox("Seat Class", ["Standard", "First", "Business"])
        total_seats = st.number_input("Total Seats", min_value=50, max_value=500, value=200, step=10)
        seats_sold  = st.number_input("Seats Sold", min_value=0, max_value=500, value=150, step=5)
        remaining   = total_seats - seats_sold
        load_factor = round(seats_sold / total_seats, 3) if total_seats > 0 else 0.0
        st.metric("Remaining Seats", remaining)
        st.metric("Load Factor", f"{load_factor:.0%}")

    with col3:
        st.markdown("#### 👤 Customer Profile")
        customer_seg    = st.selectbox("Customer Segment", ["Leisure", "Business", "Student", "Senior"])
        loyalty         = st.selectbox("Loyalty Status", ["None", "Silver", "Gold", "Platinum"])
        booking_freq    = st.slider("Booking Frequency (per quarter)", 0, 20, 4)
        avg_spend       = st.number_input("Avg Spend (£)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
        booking_channel = st.selectbox("Booking Channel", ["Web", "Mobile", "Station", "Agent"])

    st.divider()
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### 💰 Pricing Context")
        base_price    = st.number_input("Base Price at Booking (£)", min_value=5.0, max_value=1000.0, value=85.0, step=5.0)
        days_before   = st.slider("Days Before Travel", 0, 180, 14)
        price_premium = st.number_input("Price Premium Multiplier", min_value=0.5, max_value=5.0, value=1.15, step=0.05)

    with col5:
        st.markdown("#### 📊 Demand Signals")
        demand_index = st.slider("Demand Index", 0.5, 3.0, 1.5, step=0.1)
        if days_before < 7:
            st.warning("Last-minute booking — expect higher prices")
        elif days_before > 60:
            st.info("Early booking — best price window")
        if load_factor > 0.85:
            st.warning("High load factor — limited availability")

    submitted = st.form_submit_button("🔍 Forecast Price", use_container_width=True)

# ── Predict ───────────────────────────────────────────────────────────────────

if submitted:
    raw = {
        "Seat_Class":               seat_class,
        "Booking_Channel":          booking_channel,
        "Origin":                   origin,
        "Destination":              destination,
        "Distance_km":              distance_km,
        "Route_Category":           route_cat,
        "Customer_Segment":         customer_seg,
        "Loyalty_Status":           loyalty,
        "Booking_Frequency_Qtr":    booking_freq,
        "Average_Spend_GBP":        avg_spend,
        "Total_Seats":              total_seats,
        "Seats_Sold_Realized":      seats_sold,
        "Remaining_Seats_Realized": remaining,
        "Demand_Index":             demand_index,
        "Base_Price_At_Booking":    base_price,
        "Days_Before_Travel":       days_before,
        "Price_Premium":            price_premium,
        "Load_Factor":              load_factor,
    }

    X = encode_input(raw)

    predictions = []
    for name, model in models.items():
        pred, low, high = predict_with_confidence(model, X)
        predictions.append({
            "model_name":           name,
            "predicted_price_gbp":  pred,
            "confidence_low_gbp":   low,
            "confidence_high_gbp":  high,
        })

    st.divider()
    st.markdown(f"### Results: {origin} → {destination} | {seat_class} | {days_before} days out")

    cols = st.columns(len(predictions))
    for i, pred in enumerate(predictions):
        with cols[i]:
            badge = '<div class="best-badge">⭐ PRODUCTION MODEL</div><br>' if pred["model_name"] == production_model else ""
            st.markdown(f"""
            <div class="price-card">
                {badge}
                <div style="color:#94a3b8; font-size:0.8rem; margin-bottom:6px;">
                    {pred['model_name'].replace('_', ' ').title()}
                </div>
                <div class="price-big">£{pred['predicted_price_gbp']:.2f}</div>
                <div class="price-range">
                    Range: £{pred['confidence_low_gbp']:.2f} – £{pred['confidence_high_gbp']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Model Comparison")
    chart_df = pd.DataFrame([
        {"Model": p["model_name"].replace("_", " ").title(), "Predicted Price (£)": p["predicted_price_gbp"]}
        for p in predictions
    ]).set_index("Model")
    st.bar_chart(chart_df, use_container_width=True)

    prod_pred  = next((p for p in predictions if p["model_name"] == production_model), predictions[0])
    prod_price = prod_pred["predicted_price_gbp"]

    st.divider()
    ca, cb, cc = st.columns(3)
    ca.metric("Production Model Price", f"£{prod_price:.2f}")
    cb.metric("Price per km", f"£{prod_price / distance_km:.3f}")
    cc.metric("Days to Travel", f"{days_before} days")

    if days_before < 7:
        st.info(f"💡 Booking 3+ weeks earlier could save approximately £{prod_price * 0.28:.2f}")
    if loyalty in ["Gold", "Platinum"]:
        discount = {"Gold": 0.07, "Platinum": 0.12}[loyalty]
        st.success(f"✅ Your {loyalty} loyalty status saved approximately £{prod_price * discount / (1 - discount):.2f}")
