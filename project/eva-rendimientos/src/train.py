"""Modelo final T4: Optuna (LightGBM) + walk-forward + SHAP.

Flujo:
1. Optuna optimiza LightGBM sobre el corte principal (train<=2020, val 2021-2022).
2. Walk-forward (3 vueltas expansivas) compara LightGBM (best params) vs XGBoost.
3. Reentrenamiento final con todo (2006-2025) → models/model.joblib.
4. SHAP sobre el test para importancia de features.

Salidas: models/model.joblib, models/metrics.json, models/feature_config.json,
models/test_predictions.csv. La inferencia (T5) reutiliza feature_config.json:
encodings, medianas de imputación y orden de columnas.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import shap
import xgboost as xgb

from features import build_features, load_clean, metrics, prepare_xy

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
N_TRIALS = 20


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "random_state": SEED, "verbose": -1, "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return float(np.sqrt(((np.asarray(y_val) - pred) ** 2).mean()))


def main() -> None:
    models_dir = (Path(__file__).resolve().parent.parent / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    m = build_features(df)

    # --- 1. Optuna sobre el corte principal ---
    splits, gmean = prepare_xy(m, train_end=2020, val_end=2022)
    Xtr, ytr, _ = splits["train"]
    Xv, yv, _ = splits["val"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, Xtr, ytr, Xv, yv), n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    best_full = {"objective": "regression", "random_state": SEED, "verbose": -1,
                 "n_jobs": -1, **best}
    print(f"Optuna mejor RMSE(val): {study.best_value:.3f} en {N_TRIALS} trials")
    print(f"Best params: {best}")

    # --- 2. Walk-forward: 3 vueltas expansivas (LightGBM vs XGBoost) ---
    folds = [(2019, 2020), (2020, 2021), (2021, 2022)]  # (train_end, test_year)
    wf = {"lightgbm": [], "xgboost": []}
    for train_end, test_year in folds:
        s2, _ = prepare_xy(m, train_end=train_end, val_end=test_year)
        X2tr, y2tr, _ = s2["train"]
        X2te, y2te, _ = s2["test"]
        lm = lgb.LGBMRegressor(**best_full).fit(X2tr, y2tr)
        xm = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, tree_method="hist")
        xm.fit(X2tr, y2tr)
        wf["lightgbm"].append({f"fold_train<={train_end}_test{test_year}":
                               metrics(y2te, lm.predict(X2te))})
        wf["xgboost"].append({f"fold_train<={train_end}_test{test_year}":
                              metrics(y2te, xm.predict(X2te))})

    # --- 3. Reentrenamiento final con todo (2006-2025) ---
    Xall, yall, _ = splits["train"]
    Xall_full = pd_concat(splits)
    final = lgb.LGBMRegressor(**best_full).fit(Xall_full[0], Xall_full[1])

    # --- 4. SHAP sobre el test principal ---
    Xte, yte, _ = splits["test"]
    explainer = shap.TreeExplainer(final)
    shap_values = explainer.shap_values(Xte)
    shap_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(Xte.columns, shap_abs), key=lambda t: -t[1])

    # --- Métricas principales (val/test) LightGBM best ---
    main_model = lgb.LGBMRegressor(**best_full).fit(Xtr, ytr)
    main_metrics = {
        "val": metrics(yv, main_model.predict(Xv)),
        "test": metrics(yte, main_model.predict(Xte)),
        "optuna_best_val_rmse": study.best_value,
        "optuna_best_params": best,
        "walk_forward": wf,
        "rf_baseline_test": {"rmse": 5.398, "mae": 2.265, "r2": 0.877},
    }
    (models_dir / "metrics.json").write_text(
        json.dumps(main_metrics, indent=2), encoding="utf-8")

    import joblib
    joblib.dump(final, models_dir / "model.joblib")
    # Formato nativo de LightGBM (portable entre versiones de sklearn/Python)
    final.booster_.save_model(str(models_dir / "model.txt"))

    # Config de inferencia reutilizable (T5)
    config = {
        "feature_columns": list(splits["train"][0].columns),
        "global_mean": gmean,
        "best_params": best,
        "groups": sorted(m["grupo_norm"].unique().tolist()),
        "ciclos": sorted(m["ciclo_norm"].unique().tolist()),
    }
    (models_dir / "feature_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    # Predicciones test para el notebook
    out = splits["test"][2][["departamento", "municipio", "cultivo", "anio", "periodo"]].copy()
    out["y_true"] = np.asarray(yte)
    out["y_pred"] = main_model.predict(Xte)
    out["y_pred_opt"] = final.predict(Xte)
    out.to_csv(models_dir / "test_predictions.csv", index=False)

    print(f"Test LightGBM: RMSE={main_metrics['test']['rmse']:.3f} "
          f"MAE={main_metrics['test']['mae']:.3f} "
          f"R2={main_metrics['test']['r2']:.4f}")
    print("Top 10 features (SHAP):")
    for name, val in importance[:10]:
        print(f"  {name}: {val:.4f}")
    for k, v in wf.items():
        r2s = [list(x.values())[0]["r2"] for x in v]
        print(f"Walk-forward {k}: R2 por vuelta = {[round(x, 4) for x in r2s]}")


def pd_concat(splits):
    import pandas as pd
    Xs = [splits[k][0] for k in ["train", "val", "test"]]
    ys = [splits[k][1] for k in ["train", "val", "test"]]
    return pd.concat(Xs), pd.concat(ys)


if __name__ == "__main__":
    main()
