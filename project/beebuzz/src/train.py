"""Train, tune, evaluate and persist the BeeBuzz classifier.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from features import FEATURES, TARGET_SPECIES, TARGET_TYPE, load, split_features_target

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "buzz_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_macro_f1(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=skf)
    return f1_score(y, pred, average="macro")


def objective_rf(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    return cv_macro_f1(model, X, y)


def main():
    df = load(str(DATA))
    X, y, _ = split_features_target(df, TARGET_SPECIES)

    classes = sorted(y.unique())
    label_map = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(label_map)

    # --- Baseline models ---
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "rf_default": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb_default": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     objective="multi:softprob", eval_metric="mlogloss",
                                     random_state=42, use_label_encoder=False),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, m in models.items():
        pred = cross_val_predict(m, X, y_enc, cv=skf)
        acc = accuracy_score(y_enc, pred)
        f1 = f1_score(y_enc, pred, average="macro")
        print(f"{name:22s} acc={acc:.3f}  macroF1={f1:.3f}", flush=True)
        results[name] = {"accuracy": acc, "macro_f1": f1}

    # --- RF tuning (20 trials) ---
    print("\nTuning RandomForest (20 trials)...", flush=True)
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(lambda t: objective_rf(t, X, y_enc), n_trials=20, show_progress_bar=False)
    best_rf_params = study_rf.best_params
    print(f"Best RF macroF1: {study_rf.best_value:.4f}", flush=True)
    print(f"Best RF params: {best_rf_params}", flush=True)

    rf_tuned = RandomForestClassifier(**best_rf_params, class_weight="balanced", random_state=42)
    pred = cross_val_predict(rf_tuned, X, y_enc, cv=skf)
    acc = accuracy_score(y_enc, pred)
    f1 = f1_score(y_enc, pred, average="macro")
    print(f"{'rf_tuned':22s} acc={acc:.3f}  macroF1={f1:.3f}", flush=True)
    results["rf_tuned"] = {"accuracy": acc, "macro_f1": f1, "best_params": best_rf_params}

    # --- Final model: pick best among all ---
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest model by CV macroF1: {best_name} ({results[best_name]['macro_f1']:.4f})")

    model_map = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "rf_default": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb_default": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     objective="multi:softprob", eval_metric="mlogloss",
                                     random_state=42, use_label_encoder=False),
        "rf_tuned": rf_tuned,
    }
    final = model_map[best_name]
    final.fit(X, y_enc)
    joblib.dump(final, MODELS / "model.joblib")

    cfg = {
        "features": FEATURES,
        "classes": classes,
        "label_map": label_map,
        "best_model": best_name,
        "metrics_cv": results,
    }
    with open(MODELS / "feature_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nBest model: {best_name}")
    print("Saved:", MODELS / "model.joblib", flush=True)

    # --- Second target: type (floral vs flight) ---
    print("\n=== Training type classifier (floral vs flight) ===")
    X2, y2, _ = split_features_target(df, TARGET_TYPE)
    classes2 = sorted(y2.unique())
    label_map2 = {c: i for c, i in zip(classes2, range(len(classes2)))}
    y2_enc = y2.map(label_map2)

    type_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    pred2 = cross_val_predict(type_model, X2, y2_enc, cv=skf)
    acc2 = accuracy_score(y2_enc, pred2)
    f1_2 = f1_score(y2_enc, pred2, average="macro")
    print(f"type (floral/flight)  acc={acc2:.3f}  macroF1={f1_2:.3f}")

    type_model.fit(X2, y2_enc)
    joblib.dump(type_model, MODELS / "model_type.joblib")
    cfg2 = {
        "features": FEATURES,
        "classes": classes2,
        "label_map": label_map2,
        "metrics": {"accuracy": acc2, "macro_f1": f1_2},
    }
    with open(MODELS / "feature_config_type.json", "w") as f:
        json.dump(cfg2, f, indent=2)
    print("Saved:", MODELS / "model_type.joblib")


if __name__ == "__main__":
    main()
