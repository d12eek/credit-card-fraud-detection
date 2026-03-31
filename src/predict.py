
import os
import sys
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── Feature engineering constants (must match feature_engineering.py) ─────────
TOP_FRAUD_COLS = ["V4", "V11", "V12", "V14", "V17"]
V_COLS         = [f"V{i}" for i in range(1, 29)]

# ── Model weights (based on ROC-AUC from evaluate.py) ─────────────────────────
#   Isolation Forest : 0.9522  → weight 0.50
#   One-Class SVM    : 0.8056  → weight 0.20
#   LOF              : 0.9487  → weight 0.30
MODEL_WEIGHTS = {
    "Isolation Forest": 0.50,
    "One-Class SVM"   : 0.20,
    "LOF"             : 0.30,
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    """Load scaler and all 3 models from models/."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    models = {
        "Isolation Forest": joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl")),
        "One-Class SVM"   : joblib.load(os.path.join(MODELS_DIR, "one_class_svm.pkl")),
        "LOF"             : joblib.load(os.path.join(MODELS_DIR, "lof.pkl")),
    }
    return {"scaler": scaler, "models": models}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler) -> np.ndarray:
    """Scale the input using the saved StandardScaler."""
    return scaler.transform(df)


def add_features(X_scaled: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Apply same feature engineering as feature_engineering.py.
    Must stay in sync with that file.
    """
    col_names   = list(X_raw.columns)
    X_df        = pd.DataFrame(X_scaled, columns=col_names)

    # 1. Log Amount
    X_df["amount_log"]    = np.log1p(X_raw["Amount"].values)

    # 2. Hour of day
    X_df["hour_of_day"]   = (X_raw["Time"].values % 86400) / 3600

    # 3. V magnitude
    v_present             = [c for c in V_COLS if c in X_df.columns]
    X_df["v_magnitude"]   = np.linalg.norm(X_df[v_present].values, axis=1)

    # 4 & 5. Top fraud V stats
    top_present           = [c for c in TOP_FRAUD_COLS if c in X_df.columns]
    X_df["top_fraud_mean"]= X_df[top_present].mean(axis=1).values
    X_df["top_fraud_std"] = X_df[top_present].std(axis=1).values

    # 6. Amount/V ratio
    X_df["amount_v_ratio"]= X_df["amount_log"] / (X_df["v_magnitude"] + 1e-6)

    return X_df.values


def normalise_score(scores: np.ndarray) -> np.ndarray:
    """
    Sigmoid normalisation — works correctly for single transactions too.
    Higher value = more anomalous = more likely fraud.
    """
    return 1 / (1 + np.exp(-scores * 3))


def ensemble_score(X_eng: np.ndarray, models: dict) -> np.ndarray:
    """
    Get weighted average anomaly score from all 3 models.
    Returns a single confidence score per transaction in [0, 1].
    Higher = more likely fraud.
    """
    combined = np.zeros(len(X_eng))

    for name, model in models.items():
        raw    = model.decision_function(X_eng)
        flipped = -raw                        # flip: high = anomalous
        normed  = normalise_score(flipped)    # sigmoid → [0, 1]
        combined += MODEL_WEIGHTS[name] * normed

    return combined                          # weighted sum in [0, 1]


def score_to_label(confidence: float, threshold: float = 0.45) -> dict:
    """
    Convert a single confidence score to a label + alert level.

    Threshold 0.55 (instead of 0.5) biases slightly toward precision
    — reduces false alarms while still catching most fraud.
    """
    label = "FRAUD" if confidence >= threshold else "NORMAL"

    if confidence >= 0.60:
      alert = "🔴 HIGH"
    elif confidence >= 0.45:
      alert = "🟠 MEDIUM"
    elif confidence >= 0.30:
      alert = "🟡 LOW"
    else:
      alert = "🟢 SAFE"

    return {
        "label"     : label,
        "confidence": round(float(confidence), 4),
        "alert"     : alert,
    }


# ── Main predict function ─────────────────────────────────────────────────────

def predict(transactions: pd.DataFrame, artifacts: dict = None) -> pd.DataFrame:
    """
    Full prediction pipeline for one or more transactions.

    Args:
        transactions : DataFrame with same columns as creditcard.csv
                       (Time, V1–V28, Amount) — NO Class column needed
        artifacts    : optional pre-loaded dict from load_artifacts()
                       (pass it in to avoid reloading on every call in the app)

    Returns:
        DataFrame with original columns + confidence, label, alert columns
    """
    if artifacts is None:
        artifacts = load_artifacts()

    scaler = artifacts["scaler"]
    models = artifacts["models"]

    # Keep raw copy for feature engineering
    X_raw    = transactions.copy()

    # Scale
    X_scaled = preprocess(X_raw, scaler)

    # Engineer features
    X_eng    = add_features(X_scaled, X_raw)

    # Ensemble score
    scores   = ensemble_score(X_eng, models)

    # Build results
    results = []
    for i, score in enumerate(scores):
        result = score_to_label(score)
        results.append(result)

    out = transactions.copy().reset_index(drop=True)
    out["confidence"] = [r["confidence"] for r in results]
    out["label"]      = [r["label"]      for r in results]
    out["alert"]      = [r["alert"]      for r in results]

    return out


# ── Quick self-test ───────────────────────────────────────────────────────────

def _self_test():
    """
    Run predict on 5 real test samples (mix of normal + fraud)
    to verify the full pipeline works end to end.
    """
    print("\n[PREDICT] Loading artifacts...")
    artifacts = load_artifacts()
    print("  ✓ Scaler and models loaded")

    # Load a few real test rows
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    # Pick 3 normal + 2 fraud samples
    normal_idx = y_test[y_test == 0].index[:3].tolist()
    fraud_idx  = y_test[y_test == 1].index[:2].tolist()
    sample_idx = normal_idx + fraud_idx

    samples     = X_test.loc[sample_idx].reset_index(drop=True)
    true_labels = y_test.loc[sample_idx].reset_index(drop=True)

    print("\n[PREDICT] Running predictions on 5 sample transactions...\n")
    results = predict(samples, artifacts)

    for i in range(len(results)):
        true  = "FRAUD" if true_labels[i] == 1 else "NORMAL"
        pred  = results.loc[i, "label"]
        conf  = results.loc[i, "confidence"]
        alert = results.loc[i, "alert"]
        match = "✓" if true == pred else "✗"
        print(f"  [{match}] True: {true:6s}  |  Pred: {pred:6s}  "
              f"|  Confidence: {conf:.4f}  |  Alert: {alert}")

    print("\n[PREDICT] Done ✓\n")


if __name__ == "__main__":
    _self_test()