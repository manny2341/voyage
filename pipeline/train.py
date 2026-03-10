"""
train.py — Full training pipeline with MLflow tracking
Trains 5 models, saves pkl files locally, logs metrics to MLflow.
Run: python3 pipeline/train.py
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROCESSED_PATH = Path("data/processed/tickets_processed.csv")
MODELS_DIR     = Path("data/processed/models")
MLFLOW_URI     = "http://localhost:5050"
EXPERIMENT     = "voyage-ticket-pricing"
TARGET         = "Ticket_Price_GBP"

MODELS = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "ridge_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=5, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42,
    ),
}

try:
    from xgboost import XGBRegressor
    MODELS["xgboost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    print("XGBoost available — added to training run")
except ImportError:
    print("XGBoost not installed — skipping")


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100
    return {
        "mae":              round(float(mae), 4),
        "rmse":             round(float(rmse), 4),
        "r2":               round(float(r2), 4),
        "within_10gbp_pct": round(float(within_10), 2),
        "within_20gbp_pct": round(float(within_20), 2),
    }


def train_and_log(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds)

    # Save model as pkl locally
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = MODELS_DIR / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Log only params + metrics to MLflow (no artifact upload)
    try:
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            run_id = run.info.run_id
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")
        run_id = "local"

    print(f"  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
          f"R²={metrics['r2']:.4f}  Within £10={metrics['within_10gbp_pct']:.1f}%")
    print(f"  Saved → {pkl_path}")
    return run_id, metrics


def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as e:
        print(f"MLflow experiment setup failed: {e} — continuing without MLflow")

    results = {}
    for name, model in MODELS.items():
        run_id, metrics = train_and_log(name, model, X_train, X_test, y_train, y_test)
        results[name] = {"run_id": run_id, "metrics": metrics}

    with open("data/processed/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'±£10%':>8}")
    print("-"*70)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<25} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
              f"{m['r2']:>8.4f} {m['within_10gbp_pct']:>7.1f}%")

    best = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    print(f"\nBest model: {best} (RMSE={results[best]['metrics']['rmse']:.2f})")
    print("Run: make promote")


if __name__ == "__main__":
    main()
