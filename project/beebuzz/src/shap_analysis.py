"""SHAP explainability for the BeeBuzz classifier.

Usage:
    python src/shap_analysis.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import FEATURES, TARGET_SPECIES, load, split_features_target

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "buzz_data.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = load(str(DATA))
    X, y, _ = split_features_target(df, TARGET_SPECIES)

    model = joblib.load(MODELS / "model.joblib")
    cfg = json.loads((MODELS / "feature_config.json").read_text())
    classes = cfg["classes"]

    # SHAP for tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Summary plot (global feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, class_names=classes, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", REPORTS / "shap_summary.png")

    # 2. Bar plot (mean absolute SHAP per feature, per class)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, class_names=classes, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", REPORTS / "shap_bar.png")

    # 3. Top features per class
    if isinstance(shap_values, list):
        for i, cls in enumerate(classes):
            mean_abs = np.abs(shap_values[i]).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            print(f"\nTop 10 features for {cls}:")
            for idx in top_idx:
                print(f"  {FEATURES[idx]:25s}  {mean_abs[idx]:.4f}")

    # 4. Force plot for first sample (saved as html)
    try:
        force = shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0],
                                feature_names=FEATURES, matplotlib=False)
        shap.save_html(str(REPORTS / "shap_force_sample0.html"), force)
        print("Saved:", REPORTS / "shap_force_sample0.html")
    except Exception as e:
        print("Force plot skipped:", e)


if __name__ == "__main__":
    main()
