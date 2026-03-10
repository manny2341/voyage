from typing import Literal
from pydantic import BaseModel, Field


class TicketFeatures(BaseModel):
    Seat_Class: Literal["Standard", "First", "Business"]
    Booking_Channel: Literal["Web", "Mobile", "Station", "Agent"]
    Origin: Literal["London", "Manchester", "Birmingham", "Edinburgh", "Bristol", "Leeds", "Glasgow"]
    Destination: Literal["Paris", "Amsterdam", "Brussels", "Frankfurt", "Lyon", "Cologne", "Rotterdam"]
    Distance_km: float = Field(..., ge=1, le=3000, description="Route distance in km")
    Route_Category: Literal["Short", "Medium", "Long"]
    Customer_Segment: Literal["Leisure", "Business", "Student", "Senior"]
    Loyalty_Status: Literal["None", "Silver", "Gold", "Platinum"]
    Booking_Frequency_Qtr: int = Field(..., ge=0, le=50)
    Average_Spend_GBP: float = Field(..., ge=0, le=5000)
    Total_Seats: int = Field(..., ge=1, le=1000)
    Seats_Sold_Realized: int = Field(..., ge=0, le=1000)
    Remaining_Seats_Realized: int = Field(..., ge=0, le=1000)
    Demand_Index: float = Field(..., ge=0.1, le=10.0)
    Base_Price_At_Booking: float = Field(..., ge=1, le=2000)
    Days_Before_Travel: int = Field(..., ge=0, le=365)
    Price_Premium: float = Field(..., ge=0.5, le=5.0)
    Load_Factor: float = Field(..., ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "Seats_Sold_Realized": 160,
                "Remaining_Seats_Realized": 40,
                "Demand_Index": 1.5,
                "Base_Price_At_Booking": 85.0,
                "Days_Before_Travel": 14,
                "Price_Premium": 1.15,
                "Load_Factor": 0.80,
            }
        }
    }
