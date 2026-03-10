"""
app.py — Streamlit UI for Voyage Ticket Price Forecasting
Port: 8501
"""

import requests
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Voyage — Ticket Price Forecasting",
    page_icon="🚄",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #94a3b8; font-size: 0.85rem;
    }
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🚄 Voyage — Ticket Price Forecasting")
st.markdown("Enter journey details below to get AI-powered price predictions across all models.")

# ── API health check ──────────────────────────────────────────────────────────
def get_api_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

api_info = get_api_models()
if not api_info.get("available_models"):
    st.error("API is not reachable. Make sure the FastAPI server is running on port 8000.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

available_models = api_info.get("available_models", [])
production_model = api_info.get("production_model", "")
st.success(f"API connected — {len(available_models)} models loaded | Production: **{production_model}**")

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # Section 1: Route
    with col1:
        st.markdown("#### 🗺️ Route Details")
        origin      = st.selectbox("Origin", ["London","Manchester","Birmingham","Edinburgh","Bristol","Leeds","Glasgow"])
        destination = st.selectbox("Destination", ["Paris","Amsterdam","Brussels","Frankfurt","Lyon","Cologne","Rotterdam"])
        distance_km = st.number_input("Distance (km)", min_value=50.0, max_value=2000.0, value=340.0, step=10.0)
        route_cat   = st.selectbox("Route Category", ["Short", "Medium", "Long"])

    # Section 2: Seat & Capacity
    with col2:
        st.markdown("#### 💺 Seat & Capacity")
        seat_class    = st.selectbox("Seat Class", ["Standard", "First", "Business"])
        total_seats   = st.number_input("Total Seats", min_value=50, max_value=500, value=200, step=10)
        seats_sold    = st.number_input("Seats Sold", min_value=0, max_value=500, value=150, step=5)
        remaining     = total_seats - seats_sold
        load_factor   = round(seats_sold / total_seats, 3) if total_seats > 0 else 0.0
        st.metric("Remaining Seats", remaining)
        st.metric("Load Factor", f"{load_factor:.0%}")

    # Section 3: Customer
    with col3:
        st.markdown("#### 👤 Customer Profile")
        customer_seg    = st.selectbox("Customer Segment", ["Leisure", "Business", "Student", "Senior"])
        loyalty         = st.selectbox("Loyalty Status", ["None", "Silver", "Gold", "Platinum"])
        booking_freq    = st.slider("Booking Frequency (per quarter)", 0, 20, 4)
        avg_spend       = st.number_input("Avg Spend (£)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
        booking_channel = st.selectbox("Booking Channel", ["Web", "Mobile", "Station", "Agent"])

    st.divider()
    col4, col5 = st.columns(2)

    # Section 4: Pricing Context
    with col4:
        st.markdown("#### 💰 Pricing Context")
        base_price      = st.number_input("Base Price at Booking (£)", min_value=5.0, max_value=1000.0, value=85.0, step=5.0)
        days_before     = st.slider("Days Before Travel", 0, 180, 14)
        price_premium   = st.number_input("Price Premium Multiplier", min_value=0.5, max_value=5.0, value=1.15, step=0.05)

    with col5:
        st.markdown("#### 📊 Demand Signals")
        demand_index    = st.slider("Demand Index", 0.5, 3.0, 1.5, step=0.1)

        # Auto-hint
        if days_before < 7:
            st.warning("Last-minute booking — expect higher prices")
        elif days_before > 60:
            st.info("Early booking — best price window")

        if load_factor > 0.85:
            st.warning("High load factor — limited availability")

    submitted = st.form_submit_button("🔍 Forecast Price", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "Seat_Class":                seat_class,
        "Booking_Channel":           booking_channel,
        "Origin":                    origin,
        "Destination":               destination,
        "Distance_km":               distance_km,
        "Route_Category":            route_cat,
        "Customer_Segment":          customer_seg,
        "Loyalty_Status":            loyalty,
        "Booking_Frequency_Qtr":     booking_freq,
        "Average_Spend_GBP":         avg_spend,
        "Total_Seats":               total_seats,
        "Seats_Sold_Realized":       seats_sold,
        "Remaining_Seats_Realized":  remaining,
        "Demand_Index":              demand_index,
        "Base_Price_At_Booking":     base_price,
        "Days_Before_Travel":        days_before,
        "Price_Premium":             price_premium,
        "Load_Factor":               load_factor,
    }

    with st.spinner("Forecasting prices across all models..."):
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
                st.stop()
            result = r.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is the server running?")
            st.stop()

    predictions = result["predictions"]
    best_model  = result["best_model"]

    st.divider()
    st.markdown(f"### Results: {origin} → {destination} | {seat_class} | {days_before} days out")

    # ── Price cards ───────────────────────────────────────────────────────────
    cols = st.columns(len(predictions))
    for i, pred in enumerate(predictions):
        with cols[i]:
            badge = '<div class="best-badge">⭐ PRODUCTION MODEL</div><br>' if pred["model_name"] == best_model else ""
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

    # ── Bar chart comparison ──────────────────────────────────────────────────
    st.markdown("#### Model Comparison")
    import pandas as pd
    chart_df = pd.DataFrame([
        {
            "Model": p["model_name"].replace("_", " ").title(),
            "Predicted Price (£)": p["predicted_price_gbp"]
        }
        for p in predictions
    ]).set_index("Model")
    st.bar_chart(chart_df, use_container_width=True)

    # ── Context insight ───────────────────────────────────────────────────────
    production_pred = next((p for p in predictions if p["model_name"] == best_model), predictions[0])
    prod_price = production_pred["predicted_price_gbp"]

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Production Model Price", f"£{prod_price:.2f}")
    col_b.metric("Price per km", f"£{prod_price / distance_km:.3f}")
    col_c.metric("Days to Travel", f"{days_before} days")

    if days_before < 7:
        saving = prod_price * 0.28
        st.info(f"💡 Booking 3+ weeks earlier could save approximately £{saving:.2f}")
    if loyalty in ["Gold", "Platinum"]:
        discount = {"Gold": 0.07, "Platinum": 0.12}[loyalty]
        st.success(f"✅ Your {loyalty} loyalty status saved approximately £{prod_price * discount / (1 - discount):.2f}")
