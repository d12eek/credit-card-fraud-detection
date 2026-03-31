import os
import time
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

# Add src/ to path so we can import model.py
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from model import get_all_models


def load_engineered() -> dict:
    """Load engineered train/test arrays from data/processed/."""
    print("[TRAIN] Loading engineered data...")

    X_train_eng = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_eng.csv")).values
    X_test_eng  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_eng.csv")).values
    y_train     = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_test      = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze().values

    print(f"  → X_train_eng : {X_train_eng.shape}")
    print(f"  → X_test_eng  : {X_test_eng.shape}")
    print(f"  → Fraud in test : {y_test.sum()} / {len(y_test)}")

    return {
        "X_train_eng": X_train_eng,
        "X_test_eng" : X_test_eng,
        "y_train"    : y_train,
        "y_test"     : y_test,
    }


def train_all(data: dict) -> dict:
    """
    Train all 3 models and return fitted models + test data for evaluate.py.

    Notes:
      - All models train on X_train_eng (normal transactions only)
      - One-Class SVM uses a 20k subsample for speed (O(n²) complexity)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    models      = get_all_models()
    fitted      = {}
    X_train     = data["X_train_eng"]

    # Subsample for SVM only — keeps training under ~3 minutes
    np.random.seed(42)
    svm_idx     = np.random.choice(len(X_train), size=min(20_000, len(X_train)), replace=False)
    X_train_svm = X_train[svm_idx]

    print(f"\n[TRAIN] Training on {X_train.shape[0]:,} normal samples "
          f"({X_train.shape[1]} features)\n")

    for name, model in models.items():

        # SVM uses subsample
        X_fit = X_train_svm if name == "One-Class SVM" else X_train

        print(f"  ▶ {name}  (fit on {X_fit.shape[0]:,} rows) ...")
        t0 = time.time()
        model.fit(X_fit)
        elapsed = time.time() - t0
        print(f"    ✓ Done in {elapsed:.1f}s")

        # Save model
        safe_name  = name.lower().replace(" ", "_").replace("-", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        print(f"    ✓ Saved → {model_path}")

        fitted[name] = model

    return {
        "models"    : fitted,
        "X_test_eng": data["X_test_eng"],
        "y_test"    : data["y_test"],
    }


def run() -> dict:
    """Full pipeline: load → train → save models."""
    data   = load_engineered()
    result = train_all(data)
    print("\n[TRAIN] All models trained and saved ✓\n")
    return result


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()