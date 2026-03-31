import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# V columns most correlated with fraud (standard finding on this dataset)
TOP_FRAUD_COLS = ["V4", "V11", "V12", "V14", "V17"]


def load_scaled() -> dict:
    """Load scaled arrays + labels from data/processed/."""
    print("[FEATURE ENG] Loading scaled data...")

    cols           = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv")).columns.tolist()
    X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    X_test_scaled  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
    y_train        = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test         = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    # Also load raw (unscaled) X_train_normal and X_test for original Time/Amount
    X_train_normal_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv"))
    X_test_raw         = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

    print(f"  → X_train_scaled : {X_train_scaled.shape}")
    print(f"  → X_test_scaled  : {X_test_scaled.shape}")

    return {
        "X_train_scaled"    : X_train_scaled,
        "X_test_scaled"     : X_test_scaled,
        "X_train_normal_raw": X_train_normal_raw,
        "X_test_raw"        : X_test_raw,
        "y_train"           : y_train,
        "y_test"            : y_test,
        "cols"              : cols,
    }


def add_features(X_scaled: pd.DataFrame, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Takes the scaled DataFrame + original raw DataFrame,
    computes new features, appends them, and returns a numpy array.
    """
    X = X_scaled.copy()

    # 1. Log-transformed Amount (from raw, before scaling)
    X["amount_log"] = np.log1p(X_raw["Amount"].values)

    # 2. Hour of day from Time column (cyclic fraud pattern)
    X["hour_of_day"] = (X_raw["Time"].values % 86400) / 3600

    # 3. L2 norm of all V1–V28 (overall anomaly magnitude)
    v_cols = [c for c in X_scaled.columns if c.startswith("V")]
    X["v_magnitude"] = np.linalg.norm(X_scaled[v_cols].values, axis=1)

    # 4 & 5. Mean + std of top fraud-correlated V features
    top_cols_present = [c for c in TOP_FRAUD_COLS if c in X_scaled.columns]
    X["top_fraud_mean"] = X_scaled[top_cols_present].mean(axis=1).values
    X["top_fraud_std"]  = X_scaled[top_cols_present].std(axis=1).values

    # 6. Ratio: log-amount vs anomaly magnitude
    X["amount_v_ratio"] = X["amount_log"] / (X["v_magnitude"] + 1e-6)

    return X.values


def engineer(data: dict) -> dict:
    """Apply feature engineering to both train and test sets."""
    print("\n[FEATURE ENG] Adding engineered features...")

    X_train_eng = add_features(data["X_train_scaled"], data["X_train_normal_raw"])
    X_test_eng  = add_features(data["X_test_scaled"],  data["X_test_raw"])

    # Build feature name list
    base_cols = data["cols"]
    new_cols  = ["amount_log", "hour_of_day", "v_magnitude",
                 "top_fraud_mean", "top_fraud_std", "amount_v_ratio"]
    all_cols  = base_cols + new_cols

    print(f"  → Features before : {len(base_cols)}")
    print(f"  → Features added  : {len(new_cols)}  {new_cols}")
    print(f"  → Features after  : {len(all_cols)}")
    print(f"  → X_train_eng     : {X_train_eng.shape}")
    print(f"  → X_test_eng      : {X_test_eng.shape}")

    return {
        "X_train_eng" : X_train_eng,
        "X_test_eng"  : X_test_eng,
        "y_train"     : data["y_train"],
        "y_test"      : data["y_test"],
        "feature_cols": all_cols,
    }


def save_engineered(eng: dict) -> None:
    """Save engineered arrays to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cols = eng["feature_cols"]

    pd.DataFrame(eng["X_train_eng"], columns=cols).to_csv(
        os.path.join(PROCESSED_DIR, "X_train_eng.csv"), index=False
    )
    pd.DataFrame(eng["X_test_eng"], columns=cols).to_csv(
        os.path.join(PROCESSED_DIR, "X_test_eng.csv"), index=False
    )
    print(f"\n[FEATURE ENG] Engineered CSVs saved → {PROCESSED_DIR}")


def run(save: bool = True) -> dict:
    """Full pipeline: load scaled → engineer → save."""
    data = load_scaled()
    eng  = engineer(data)

    if save:
        save_engineered(eng)

    print("\n[FEATURE ENG] Done ✓\n")
    return eng


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()