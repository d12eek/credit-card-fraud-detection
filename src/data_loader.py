
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH      = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data(filepath: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    print(f"\n[DATA LOADER] Reading file: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  → Raw shape          : {df.shape}")
    print(f"  → Null values total  : {df.isnull().sum().sum()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the single all-NaN row (same as notebook: df.dropna()).
    Cast Class to int for cleaner downstream use.
    """
    before = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    df_clean["Class"] = df_clean["Class"].astype(int)
    after = len(df_clean)

    print(f"\n[DATA LOADER] Cleaning")
    print(f"  → Rows before dropna : {before}")
    print(f"  → Rows after  dropna : {after}  (dropped {before - after} row(s))")

    fraud_count  = df_clean["Class"].sum()
    total        = len(df_clean)
    fraud_pct    = fraud_count / total * 100
    print(f"\n  → Class distribution")
    print(f"     Normal : {total - fraud_count:,}  ({100 - fraud_pct:.3f}%)")
    print(f"     Fraud  : {fraud_count:,}  ({fraud_pct:.3f}%)")

    return df_clean


def split_data(
    df_clean: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Stratified 70/30 train–test split (same as notebook).
    Also creates X_train_normal (only legitimate transactions)
    which is what anomaly models are trained on.
    """
    X = df_clean.drop(columns=["Class"])
    y = df_clean["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keeps fraud ratio the same in both splits
    )

    # Anomaly models train ONLY on normal transactions
    X_train_normal = X_train[y_train == 0]

    print(f"\n[DATA LOADER] Train–Test Split  (test_size={test_size}, stratified)")
    print(f"  → X_train        : {X_train.shape}")
    print(f"  → X_train_normal : {X_train_normal.shape}  (fraud-free, for model fitting)")
    print(f"  → X_test         : {X_test.shape}")
    print(f"  → Fraud in test  : {y_test.sum()} / {len(y_test)}")

    return {
        "X_train"        : X_train,
        "X_test"         : X_test,
        "y_train"        : y_train,
        "y_test"         : y_test,
        "X_train_normal" : X_train_normal,
    }


def save_processed(splits: dict) -> None:
    """Save all splits to data/processed/ as CSV files."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    splits["X_train"].to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"),        index=False)
    splits["X_test"].to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"),         index=False)
    splits["y_train"].to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"),        index=False)
    splits["y_test"].to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"),         index=False)
    splits["X_train_normal"].to_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv"), index=False)

    print(f"\n[DATA LOADER] Processed splits saved → {PROCESSED_DIR}")


def run(filepath: str = RAW_PATH, save: bool = True) -> dict:
    """
    Full pipeline: load → clean → split → (optionally save).
    Returns the splits dict for direct use by other modules.
    """
    df        = load_data(filepath)
    df_clean  = clean_data(df)
    splits    = split_data(df_clean)

    if save:
        save_processed(splits)

    print("\n[DATA LOADER] Done ✓\n")
    return splits


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()