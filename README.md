# Voyage — Ticket Price Forecasting System

A containerized machine learning system that predicts railway ticket prices using multiple ML models, served through a FastAPI backend and Streamlit web interface, with MLflow experiment tracking.

---

## Features

- Predicts ticket prices using 5 ML models side by side
- Confidence range for every prediction (low / high estimate)
- Interactive web UI with grouped input sections
- Bar chart comparison across all models
- Smart booking hints (last-minute pricing, loyalty discounts)
- MLflow experiment tracking and model registry
- Fully containerized with Docker

---

## Models

| Model | Description |
|---|---|
| Linear Regression | Fast baseline with feature scaling |
| Ridge Regression | Regularised linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting regressor |
| XGBoost | High-performance gradient boosting |

---

## Architecture

```
User → Streamlit UI (8501) → FastAPI API (8000) → Trained Models (pkl)
                                        ↓
                               MLflow Tracking (5050)
```

---

## Quick Start

**Run with Docker:**
```bash
git clone https://github.com/manny2341/voyage
cd voyage
make docker
```

**Train models (first time):**
```bash
make train
make promote
make docker
```

**Access:**
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow Dashboard | http://localhost:5050 |

---

## Run Without Docker

```bash
make install       # Install dependencies
make pipeline      # Generate data → preprocess → train → promote
make mlflow        # Start MLflow (terminal 1)
make serve         # Start API (terminal 2)
make ui            # Start UI (terminal 3)
```

---

## Input Features

| Feature | Description |
|---|---|
| Seat Class | Standard / First / Business |
| Route | Origin, Destination, Distance, Category |
| Customer | Segment, Loyalty Status, Booking Frequency |
| Pricing | Base Price, Days Before Travel, Demand Index |
| Capacity | Total Seats, Seats Sold, Load Factor |

---

## Tech Stack

- **ML:** Scikit-learn, XGBoost
- **API:** FastAPI, Uvicorn, Pydantic
- **UI:** Streamlit
- **Tracking:** MLflow
- **Infrastructure:** Docker, Docker Compose

---

## Author

[@manny2341](https://github.com/manny2341)
