"""Baselines T3: media histórica + LinearRegression + RandomForest.

Validación temporal: train 2006-2020, val 2021-2022, test 2023-2025.
Guarda models/baseline_metrics.json y models/baseline_test_predictions.csv
(RF) para comparar contra el modelo final T4.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from features import build_features, load_clean, metrics, prepare_xy


def main() -> None:
    models_dir = (Path(__file__).resolve().parent.parent / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    m = build_features(df)
    splits, global_mean = prepare_xy(m)
    Xtr, ytr, _ = splits["train"]
    results = {"global_mean": global_mean,
               "features": list(Xtr.columns),
               "splits": {k: {"n": len(v[1])} for k, v in splits.items()}}

    # 1. Media histórica por cultivo (solo train) → grupo → global
    tr, _, tr_meta = splits["train"]
    cult_mean = tr_meta.groupby("cultivo_norm")["rendimiento_clipped"].mean().to_dict()
    grup_mean = tr_meta.groupby("grupo_norm")["rendimiento_clipped"].mean().to_dict()
    results["media_historica"] = {}
    for name in ["val", "test"]:
        _, y, meta = splits[name]
        pred = (meta["cultivo_norm"].map(cult_mean)
                .fillna(meta["grupo_norm"].map(grup_mean))
                .fillna(global_mean).to_numpy())
        results["media_historica"][name] = metrics(y, pred)

    # 2. LinearRegression (numéricos estandarizados, dummies tal cual)
    scaler = StandardScaler()
    Xtr_lin = scaler.fit_transform(Xtr)
    lin = LinearRegression().fit(Xtr_lin, ytr)
    results["linear"] = {}
    for name in ["val", "test"]:
        X, y, _ = splits[name]
        results["linear"][name] = metrics(y, lin.predict(scaler.transform(X)))

    # 3. RandomForest
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    results["random_forest"] = {}
    for name in ["val", "test"]:
        X, y, _ = splits[name]
        results["random_forest"][name] = metrics(y, rf.predict(X))

    (models_dir / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    Xte, yte, meta = splits["test"]
    out = meta[["departamento", "municipio", "cultivo", "anio", "periodo"]].copy()
    out["y_true"] = np.asarray(yte)
    out["y_pred_media"] = (meta["cultivo_norm"].map(cult_mean)
                           .fillna(meta["grupo_norm"].map(grup_mean))
                           .fillna(global_mean).to_numpy())
    out["y_pred_rf"] = rf.predict(Xte)
    out.to_csv(models_dir / "baseline_test_predictions.csv", index=False)

    for model in ["media_historica", "linear", "random_forest"]:
        print(f"{model}: test RMSE={results[model]['test']['rmse']:.3f} "
              f"MAE={results[model]['test']['mae']:.3f} "
              f"R2={results[model]['test']['r2']:.4f} "
              f"MAPE={results[model]['test']['mape_pct']:.1f}%")
    print(f"Métricas -> {models_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
