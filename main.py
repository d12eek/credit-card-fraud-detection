# main.py
import os
import sys
import runpy

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

SRC = os.path.join(os.path.dirname(__file__), "src")

def run(script_name):
    runpy.run_path(os.path.join(SRC, script_name), run_name="__main__")

def main():
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — Full Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    run("data_loader.py")

    print("\n[2/5] Preprocessing...")
    run("preprocessing.py")

    print("\n[3/5] Feature Engineering...")
    run("feature_engineering.py")

    print("\n[4/5] Training models...")
    run("train.py")

    print("\n[5/5] Evaluating models...")
    run("evaluate.py")

    print("\n" + "=" * 60)
    print("  Pipeline complete ✓")
    print("  → Models saved in:  models/")
    print("  → Plots  saved in:  models/plots/")
    print("  → Run the app with: streamlit run app/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()