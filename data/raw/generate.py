"""
generate.py — Creates a realistic synthetic railway ticket dataset
Run once: python data/raw/generate.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 10000

SEAT_CLASSES      = ["Standard", "First", "Business"]
BOOKING_CHANNELS  = ["Web", "Mobile", "Station", "Agent"]
ORIGINS           = ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol", "Leeds", "Glasgow"]
DESTINATIONS      = ["Paris", "Amsterdam", "Brussels", "Frankfurt", "Lyon", "Cologne", "Rotterdam"]
ROUTE_CATEGORIES  = ["Short", "Medium", "Long"]
CUSTOMER_SEGMENTS = ["Leisure", "Business", "Student", "Senior"]
LOYALTY_STATUSES  = ["None", "Silver", "Gold", "Platinum"]

seat_class       = np.random.choice(SEAT_CLASSES, N, p=[0.65, 0.20, 0.15])
booking_channel  = np.random.choice(BOOKING_CHANNELS, N, p=[0.50, 0.30, 0.12, 0.08])
origin           = np.random.choice(ORIGINS, N)
destination      = np.random.choice(DESTINATIONS, N)
route_category   = np.random.choice(ROUTE_CATEGORIES, N, p=[0.35, 0.40, 0.25])
customer_segment = np.random.choice(CUSTOMER_SEGMENTS, N, p=[0.45, 0.30, 0.15, 0.10])
loyalty_status   = np.random.choice(LOYALTY_STATUSES, N, p=[0.40, 0.25, 0.20, 0.15])

distance_map = {"Short": (50, 200), "Medium": (200, 600), "Long": (600, 1500)}
distance_km  = np.array([
    np.random.uniform(*distance_map[rc]) for rc in route_category
])

days_before_travel        = np.random.randint(1, 180, N)
booking_frequency_qtr     = np.random.randint(0, 20, N)
average_spend_gbp         = np.random.uniform(20, 500, N).round(2)
total_seats               = np.random.choice([100, 150, 200, 250, 300], N)
load_factor               = np.random.uniform(0.3, 1.0, N).round(3)
seats_sold                = (total_seats * load_factor).astype(int)
remaining_seats           = total_seats - seats_sold
demand_index              = np.random.uniform(0.5, 3.0, N).round(3)

# Base price driven by distance and seat class
class_multiplier = {"Standard": 1.0, "First": 2.2, "Business": 1.7}
base_price = (
    15
    + distance_km * 0.12
    + np.array([class_multiplier[c] for c in seat_class]) * 20
    + np.random.normal(0, 8, N)
).round(2)
base_price = np.clip(base_price, 10, 600)

price_premium = np.where(
    days_before_travel < 7,  1.4,
    np.where(days_before_travel < 30, 1.15, 1.0)
) + np.random.uniform(-0.05, 0.05, N)

# Final ticket price — the target
loyalty_discount = {"None": 0, "Silver": 0.03, "Gold": 0.07, "Platinum": 0.12}
ticket_price = (
    base_price
    * price_premium
    * demand_index * 0.3
    + base_price * 0.7
    - np.array([loyalty_discount[l] for l in loyalty_status]) * base_price
    + np.random.normal(0, 5, N)
).round(2)
ticket_price = np.clip(ticket_price, 10, 900)

df = pd.DataFrame({
    "Seat_Class":                seat_class,
    "Booking_Channel":           booking_channel,
    "Origin":                    origin,
    "Destination":               destination,
    "Distance_km":               distance_km.round(1),
    "Route_Category":            route_category,
    "Customer_Segment":          customer_segment,
    "Loyalty_Status":            loyalty_status,
    "Booking_Frequency_Qtr":     booking_frequency_qtr,
    "Average_Spend_GBP":         average_spend_gbp,
    "Total_Seats":               total_seats,
    "Seats_Sold_Realized":       seats_sold,
    "Remaining_Seats_Realized":  remaining_seats,
    "Demand_Index":              demand_index,
    "Base_Price_At_Booking":     base_price,
    "Days_Before_Travel":        days_before_travel,
    "Price_Premium":             price_premium.round(3),
    "Load_Factor":               load_factor,
    "Ticket_Price_GBP":          ticket_price,
})

df.to_csv("data/raw/tickets.csv", index=False)
print(f"Dataset saved — {len(df)} rows, {df.columns.tolist()}")
print(df["Ticket_Price_GBP"].describe().round(2))
