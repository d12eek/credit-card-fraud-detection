from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# ── Fraud rate from data_loader output (492 / 284807) ─────────────────────────
FRAUD_RATE = 492 / 284807   # ≈ 0.001728  (actual rate in full dataset)


def get_isolation_forest() -> IsolationForest:
    """
    Isolation Forest — primary anomaly detector.

    Improvements vs notebook:
      n_estimators : 300  (was 200) — more trees = more stable scores
      contamination: FRAUD_RATE    (was 0.002) — matches actual fraud ratio
      max_samples  : 'auto'        — uses min(256, n_samples) per tree
    """
    return IsolationForest(
        n_estimators=300,
        contamination=FRAUD_RATE,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )


def get_ocsvm() -> OneClassSVM:
    """
    One-Class SVM — secondary anomaly detector.

    Improvements vs notebook:
      nu   : FRAUD_RATE  (was 0.002) — upper bound on fraction of outliers
      gamma: 'scale'     (same)      — auto-scales to 1/(n_features * X.var())
    """
    return OneClassSVM(
        kernel="rbf",
        nu=FRAUD_RATE,
        gamma="scale",
    )


def get_lof() -> LocalOutlierFactor:
    """
    Local Outlier Factor — tertiary anomaly detector (novelty mode).

    Improvements vs notebook:
      n_neighbors  : 35   (was 20) — larger neighborhood = more robust scoring
      contamination: FRAUD_RATE   (was 0.002)
      novelty      : True (same)  — required to use predict() on new data
    """
    return LocalOutlierFactor(
        n_neighbors=35,
        contamination=FRAUD_RATE,
        novelty=True,
        n_jobs=-1,
    )


def get_all_models() -> dict:
    """Return all 3 models as a named dict — used by train.py."""
    return {
        "Isolation Forest": get_isolation_forest(),
        "One-Class SVM"   : get_ocsvm(),
        "LOF"             : get_lof(),
    }


# ── Run standalone — just prints model configs ─────────────────────────────────
if __name__ == "__main__":
    models = get_all_models()
    print("\n[MODEL] Configured models\n")
    for name, m in models.items():
        print(f"  {name}")
        print(f"    {m.get_params()}\n")
    print(f"[MODEL] Fraud rate used for contamination/nu : {FRAUD_RATE:.6f}")
    print("\n[MODEL] Done ✓\n")