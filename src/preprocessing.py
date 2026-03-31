
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")


def load_splits() -> dict:
    """Load the processed CSVs saved by data_loader.py."""
    print("[PREPROCESSOR] Loading processed splits...")

    X_train_normal = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv"))
    X_train        = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test         = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train        = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test         = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"  → X_train_normal : {X_train_normal.shape}")
    print(f"  → X_train        : {X_train.shape}")
    print(f"  → X_test         : {X_test.shape}")

    return {
        "X_train_normal" : X_train_normal,
        "X_train"        : X_train,
        "X_test"         : X_test,
        "y_train"        : y_train,
        "y_test"         : y_test,
    }


def scale(splits: dict) -> dict:
    """
    Fit StandardScaler on X_train_normal only (same as notebook).
    Transform X_train_normal and X_test with the fitted scaler.

    Why fit on normal only?
      Anomaly models learn what 'normal' looks like.
      Including fraud signals during scaling would leak information.
    """
    print("\n[PREPROCESSOR] Fitting StandardScaler on X_train_normal...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(splits["X_train_normal"])
    X_test_scaled  = scaler.transform(splits["X_test"])

    print(f"  → X_train_scaled : {X_train_scaled.shape}")
    print(f"  → X_test_scaled  : {X_test_scaled.shape}")
    print(f"  → Mean  (first 3 features): {scaler.mean_[:3].round(4)}")
    print(f"  → Scale (first 3 features): {scaler.scale_[:3].round(4)}")

    return {
        "scaler"         : scaler,
        "X_train_scaled" : X_train_scaled,
        "X_test_scaled"  : X_test_scaled,
        "y_train"        : splits["y_train"],
        "y_test"         : splits["y_test"],
    }


def save_scaler(scaler: StandardScaler) -> None:
    """Persist the fitted scaler so predict.py can reuse it."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[PREPROCESSOR] Scaler saved → {SCALER_PATH}")


def save_scaled(scaled: dict) -> None:
    """Save scaled arrays to data/processed/ as CSV."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    cols = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv")).columns.tolist()

    pd.DataFrame(scaled["X_train_scaled"], columns=cols).to_csv(
        os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False
    )
    pd.DataFrame(scaled["X_test_scaled"], columns=cols).to_csv(
        os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False
    )
    print(f"[PREPROCESSOR] Scaled CSVs saved → {PROCESSED_DIR}")


def load_scaler() -> StandardScaler:
    """Load a previously saved scaler (used by predict.py)."""
    scaler = joblib.load(SCALER_PATH)
    print(f"[PREPROCESSOR] Scaler loaded from {SCALER_PATH}")
    return scaler


def run(save: bool = True) -> dict:
    """Full pipeline: load splits → scale → save scaler + scaled arrays."""
    splits = load_splits()
    scaled = scale(splits)

    if save:
        save_scaler(scaled["scaler"])
        save_scaled(scaled)

    print("\n[PREPROCESSOR] Done ✓\n")
    return scaled


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()