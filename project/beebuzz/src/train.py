"""Train, tune, evaluate and persist the BeeBuzz classifier.

Protocol (honest):
- Stratified holdout 80/20 (seed 42) reserved BEFORE any tuning. Untouched until final eval.
- 5-fold stratified CV + Optuna tuning (20 trials) on the TRAIN 80% only.
- Final eval: best model on the untouched holdout 20%.
- Reports: confusion matrices, per-species precision/recall, Brier score + ECE + calibration curve.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

from features import FEATURES, TARGET_SPECIES, TARGET_TYPE, load, split_features_target

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "buzz_data.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

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


def ece_score(y_true, y_proba, n_bins=10):
    """Expected Calibration Error (multiclass, max-prob)."""
    y_true = np.asarray(y_true)
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    acc = (pred == y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def main():
    df = load(str(DATA))
    X, y, _ = split_features_target(df, TARGET_SPECIES)

    classes = sorted(y.unique())
    label_map = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(label_map)

    # --- Untouched holdout BEFORE tuning ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    print(f"train: {X_tr.shape}, holdout: {X_te.shape} (untouched until final eval)")

    # --- Baselines + candidates: CV on TRAIN only ---
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "rf_default": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb_default": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     objective="multi:softprob", eval_metric="mlogloss",
                                     random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, m in models.items():
        pred = cross_val_predict(m, X_tr, y_tr, cv=skf)
        acc = accuracy_score(y_tr, pred)
        f1 = f1_score(y_tr, pred, average="macro")
        print(f"{name:22s} (train CV) acc={acc:.3f}  macroF1={f1:.3f}", flush=True)
        results[name] = {"accuracy_cv_train": acc, "macro_f1_cv_train": f1}

    # --- RF tuning on TRAIN only ---
    print("\nTuning RandomForest (20 trials, train only)...", flush=True)
    study_rf = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42))
    study_rf.optimize(lambda t: objective_rf(t, X_tr, y_tr), n_trials=20, show_progress_bar=False)
    best_rf_params = study_rf.best_params
    print(f"Best RF (train CV) macroF1: {study_rf.best_value:.4f}", flush=True)

    rf_tuned = RandomForestClassifier(**best_rf_params, class_weight="balanced", random_state=42)
    pred = cross_val_predict(rf_tuned, X_tr, y_tr, cv=skf)
    acc = accuracy_score(y_tr, pred)
    f1 = f1_score(y_tr, pred, average="macro")
    print(f"{'rf_tuned':22s} (train CV) acc={acc:.3f}  macroF1={f1:.3f}", flush=True)
    results["rf_tuned"] = {"accuracy_cv_train": acc, "macro_f1_cv_train": f1, "best_params": best_rf_params}

    # --- Pick best by train CV, evaluate ONCE on holdout ---
    cand = {k: v["macro_f1_cv_train"] for k, v in results.items()}
    best_name = max(cand, key=cand.get)
    print(f"\nBest by train CV: {best_name} ({cand[best_name]:.4f})")

    model_map = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "rf_default": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb_default": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     objective="multi:softprob", eval_metric="mlogloss",
                                     random_state=42),
        "rf_tuned": rf_tuned,
    }
    final = model_map[best_name]
    final.fit(X_tr, y_tr)

    # Holdout eval
    y_pred = final.predict(X_te)
    y_proba = final.predict_proba(X_te) if hasattr(final, "predict_proba") else None
    acc_h = accuracy_score(y_te, y_pred)
    f1_h = f1_score(y_te, y_pred, average="macro")
    print(f"HOLDOUT ({best_name}): acc={acc_h:.3f} macroF1={f1_h:.3f} (n={len(y_te)})")
    results[best_name]["accuracy_holdout"] = acc_h
    results[best_name]["macro_f1_holdout"] = f1_h

    print("\nClassification report (holdout):")
    rep = classification_report(y_te, y_pred, target_names=classes, output_dict=False)
    print(rep)
    rep_dict = classification_report(y_te, y_pred, target_names=classes, output_dict=True)

    # Confusion matrix (holdout)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — holdout (n={len(y_te)}, {best_name})")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix_holdout.png", dpi=150)
    plt.close()
    print("saved confusion_matrix_holdout.png")

    # Calibration: Brier + ECE + curve (one-vs-rest mean)
    cal = {}
    if y_proba is not None:
        Yb = label_binarize(y_te, classes=list(range(len(classes))))
        brier = np.mean([brier_score_loss(Yb[:, i], y_proba[:, i]) for i in range(len(classes))])
        ece = ece_score(y_te.values, y_proba)
        cal = {"brier_ovr_mean": float(brier), "ece_maxprob": ece}
        print(f"Brier (OvR mean): {brier:.4f} | ECE: {ece:.4f}")
        # reliability curve on max-prob
        prob_true, prob_pred = calibration_curve(
            (y_proba.argmax(axis=1) == y_te.values).astype(int),
            y_proba.max(axis=1), n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred, prob_true, marker="o", label="model")
        ax.plot([0, 1], [0, 1], "--", label="perfect")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"Calibration (max-prob) — holdout, {best_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS / "calibration_holdout.png", dpi=150)
        plt.close()
        print("saved calibration_holdout.png")
        print("NOTE: predict_proba of RF is vote frequency, not calibrated probability. See README.")

    # Persist best model (fit on train; holdout stays untouched for future evals? No—save train-fit)
    joblib.dump(final, MODELS / "model.joblib")
    cfg = {
        "features": FEATURES,
        "classes": classes,
        "label_map": label_map,
        "best_model": best_name,
        "metrics": results,
        "classification_report_holdout": rep_dict,
        "calibration_holdout": cal,
        "protocol": "stratified 80/20 holdout (seed 42) reserved before tuning; 5-fold CV + Optuna(20) on train only; final eval once on holdout",
    }
    with open(MODELS / "feature_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved: {MODELS / 'model.joblib'}")

    # --- Second target: type (floral/flight), same honest protocol ---
    print("\n=== Type classifier (floral vs flight), same protocol ===")
    X2, y2, _ = split_features_target(df, TARGET_TYPE)
    classes2 = sorted(y2.unique())
    lm2 = {c: i for i, c in enumerate(classes2)}
    y2_enc = y2.map(lm2)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2_enc, test_size=0.2, stratify=y2_enc, random_state=42)
    tm = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    tm.fit(X2_tr, y2_tr)
    acc2 = accuracy_score(y2_te, tm.predict(X2_te))
    print(f"type holdout acc={acc2:.3f} (n={len(y2_te)})")
    print("WARNING: filename prefix B_/V_ perfectly predicts type. 99% may reflect recording setup, not biology.")
    joblib.dump(tm, MODELS / "model_type.joblib")
    with open(MODELS / "feature_config_type.json", "w") as f:
        json.dump({"features": FEATURES, "classes": classes2, "label_map": lm2,
                   "metrics": {"accuracy_holdout": acc2}}, f, indent=2)


if __name__ == "__main__":
    main()
