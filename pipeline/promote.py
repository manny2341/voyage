"""
promote.py — Reads training results, picks best model by RMSE,
saves active_model.json so the API knows which model is Production.
Run: python3 pipeline/promote.py
"""

import json
from pathlib import Path

RESULTS_PATH = Path("data/processed/training_results.json")
ACTIVE_PATH  = Path("data/processed/active_model.json")


def main():
    if not RESULTS_PATH.exists():
        print("No training_results.json found. Run pipeline/train.py first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best      = results[best_name]

    print(f"\nModel Rankings by RMSE:")
    print("-" * 50)
    for name, res in sorted(results.items(), key=lambda x: x[1]["metrics"]["rmse"]):
        m = res["metrics"]
        marker = " ← PRODUCTION" if name == best_name else ""
        print(f"  {name:<25} RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}{marker}")

    active = {
        "production_model": best_name,
        "all_models": list(results.keys()),
        "rmse": best["metrics"]["rmse"],
        "r2": best["metrics"]["r2"],
        "metrics": best["metrics"],
    }

    with open(ACTIVE_PATH, "w") as f:
        json.dump(active, f, indent=2)

    print(f"\nProduction model set to: {best_name}")
    print(f"Saved → {ACTIVE_PATH}")


if __name__ == "__main__":
    main()
