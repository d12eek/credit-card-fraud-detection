import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves plots without display issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PLOTS_DIR     = os.path.join(MODELS_DIR, "plots")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_data() -> tuple:
    """Load engineered test set and labels."""
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_eng.csv")).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze().values
    return X_test, y_test


def load_models() -> dict:
    """Load all saved .pkl models from models/."""
    model_files = {
        "Isolation Forest": "isolation_forest.pkl",
        "One-Class SVM"   : "one_class_svm.pkl",
        "LOF"             : "lof.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        models[name] = joblib.load(path)
        print(f"  ✓ Loaded {name}")
    return models


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Find threshold that maximises F2-score on the PR curve.
    F2 weights recall twice as much as precision — right for fraud detection.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f2_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        if (4 * p + r) > 0:
            f2 = (5 * p * r) / (4 * p + r)
        else:
            f2 = 0.0
        f2_scores.append(f2)
    best_idx       = int(np.argmax(f2_scores))
    best_threshold = thresholds[best_idx]
    return best_threshold


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Run full evaluation for one model. Returns a metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Raw anomaly scores (higher = more anomalous)
    raw_scores = model.decision_function(X_test)
    scores     = -raw_scores          # flip: high score → more likely fraud

    # ── Optimal threshold via PR curve (improvement over notebook) ────────────
    best_thresh = find_best_threshold(y_test, scores)
    preds       = (scores >= best_thresh).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    roc_auc  = roc_auc_score(y_test, scores)
    pr_auc   = average_precision_score(y_test, scores)
    f2       = fbeta_score(y_test, preds, beta=2, zero_division=0)

    print(f"\n  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  F2-Score : {f2:.4f}   (recall-weighted)")
    print(f"\n  Classification Report (optimal threshold = {best_thresh:.4f})")
    print(classification_report(y_test, preds,
                                 target_names=["Normal", "Fraud"],
                                 zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Confusion Matrix")
    safe = name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe}_confusion.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curve")
    ax.legend()
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe}_roc.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── PR Curve ──────────────────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{name} — Precision-Recall Curve")
    ax.legend()
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe}_pr.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Score Distribution ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_test == 0], bins=50, alpha=0.6, density=True, label="Normal")
    ax.hist(scores[y_test == 1], bins=50, alpha=0.6, density=True, label="Fraud")
    ax.axvline(best_thresh, color="red", linestyle="--", label=f"Threshold={best_thresh:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{name} — Score Distribution")
    ax.legend()
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe}_score_dist.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved → {PLOTS_DIR}")

    return {
        "model"    : name,
        "ROC-AUC"  : round(roc_auc, 4),
        "PR-AUC"   : round(pr_auc, 4),
        "F2-Score" : round(f2, 4),
        "Threshold": round(float(best_thresh), 4),
        "Fraud Caught" : int(cm[1, 1]),
        "Fraud Missed" : int(cm[1, 0]),
        "False Alarms" : int(cm[0, 1]),
    }


def print_summary(results: list) -> None:
    """Print a clean side-by-side comparison table."""
    print(f"\n{'='*60}")
    print("  SUMMARY — All Models")
    print(f"{'='*60}")
    df = pd.DataFrame(results).set_index("model")
    print(df.to_string())
    print(f"\n  Total fraud in test set : 148")
    print(f"{'='*60}\n")


def run() -> list:
    """Full evaluation pipeline."""
    print("\n[EVALUATE] Loading test data and models...")
    X_test, y_test = load_test_data()
    models         = load_models()

    results = []
    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    print_summary(results)
    print("[EVALUATE] Done ✓\n")
    return results


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()