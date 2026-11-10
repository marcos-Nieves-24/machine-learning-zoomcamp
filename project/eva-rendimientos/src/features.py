"""Features temporales T3 (sin fuga) + split temporal.

Lags y medias móviles se construyen por serie (municipio × cultivo) ordenada
en el tiempo, usando shift() hacia atrás: solo datos pasados, nunca el presente.
El target encoding se calcula solo sobre el train (2006–2020) y se aplica a
validación/test con suavizado (m=10) — sin fuga del futuro.

Excluidos por diseño (fuga): area_cosechada_ha, produccion_t, ratio_cosecha
(son resultado de la cosecha; a la hora de sembrar no se conocen).
Filtrados: no modelables, inconsistentes, cultivos raros (<100 filas).
"""
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = 2020
VAL_END = 2022  # train <=2020, val 2021-2022, test 2023-2025
SMOOTH_M = 10.0


def load_clean() -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent / "data"
    path = base / "eva_clean" / "eva_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Re-ejecutar: python src/clean.py")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega lags/rolling por serie. Solo columnas nuevas, sin tocar el resto."""
    m = df[df["target_modelable"] & ~df["inconsistente"]
          & ~df["cultivo_raro"]].copy()
    m["po"] = m["periodo_tipo"].map({"A": 0, "B": 1, "anual": 0})
    m = m.sort_values(["cod_mun", "cultivo_norm", "anio", "po"]).reset_index(drop=True)
    g = m.groupby(["cod_mun", "cultivo_norm"], sort=False)
    r = m["rendimiento_clipped"]
    m["r_lag1"] = g["rendimiento_clipped"].shift(1)
    m["r_lag2"] = g["rendimiento_clipped"].shift(2)
    # Rolling sobre la serie desplazada: solo pasado.
    m["r_roll3"] = (g["rendimiento_clipped"]
                    .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    m["r_roll5"] = (g["rendimiento_clipped"]
                    .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()))
    m["hist_len"] = g.cumcount()  # observaciones previas en la serie (0 = primera)
    m["area_sembrada_log"] = np.log1p(m["area_sembrada_ha"].clip(lower=0))
    m["is_B"] = (m["periodo_tipo"] == "B").astype(int)
    m["is_anual"] = (m["periodo_tipo"] == "anual").astype(int)
    return m


def smoothed_target_encoding(train: pd.DataFrame, col: str,
                               global_mean: float) -> dict:
    """Media por categoría con suavizado m=SMOOTH_M, calculada solo en train."""
    stats = train.groupby(col)["rendimiento_clipped"].agg(["mean", "count"])
    return ((stats["count"] * stats["mean"] + SMOOTH_M * global_mean)
            / (stats["count"] + SMOOTH_M)).to_dict()


def prepare_xy(m: pd.DataFrame, train_end: int = TRAIN_END,
               val_end: int = VAL_END):
    """Split temporal + encodings (solo train) + imputación. Devuelve X/y por corte.

    train: anio <= train_end | val: (train_end, val_end] | test: > val_end.
    Para walk-forward se invoca con train_end/val_end variables por vuelta.
    """
    train_mask = m["anio"] <= train_end
    val_mask = (m["anio"] > train_end) & (m["anio"] <= val_end)
    test_mask = m["anio"] > val_end
    tr = m[train_mask]
    global_mean = float(tr["rendimiento_clipped"].mean())
    enc_cols = ["cultivo_norm", "municipio_norm", "cod_dep", "grupo_norm"]
    enc_maps = {c: smoothed_target_encoding(tr, c, global_mean) for c in enc_cols}
    grupo_median = tr.groupby("grupo_norm")["r_lag1"].median().to_dict()

    def build_X(part: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=part.index)
        X["area_sembrada_log"] = part["area_sembrada_log"]
        X["is_B"] = part["is_B"]
        X["is_anual"] = part["is_anual"]
        X["hist_len"] = part["hist_len"]
        for lag in ["r_lag1", "r_lag2", "r_roll3", "r_roll5"]:
            X[lag] = part[lag]
        for c in enc_cols:
            X[c + "_te"] = part[c].map(enc_maps[c]).fillna(global_mean)
        # Imputación de lags con mediana del grupo (solo train) → global
        grupo_med = part["grupo_norm"].map(grupo_median).fillna(global_mean)
        for lag in ["r_lag1", "r_lag2", "r_roll3", "r_roll5"]:
            X[lag] = X[lag].fillna(grupo_med)
        # Dummies de baja cardinalidad (nombres seguros para LightGBM: sin comas)
        for gval in sorted(tr["grupo_norm"].unique()):
            safe = gval.replace(",", "").replace(" ", "_")
            X["grupo__" + safe] = (part["grupo_norm"] == gval).astype(int)
        for cval in sorted(tr["ciclo_norm"].unique()):
            X["ciclo__" + cval] = (part["ciclo_norm"] == cval).astype(int)
        return X

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask),
                       ("test", test_mask)]:
        part = m[mask]
        splits[name] = (build_X(part),
                        part["rendimiento_clipped"].reset_index(drop=True),
                        part.reset_index(drop=True))
    return splits, global_mean


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None))) * 100)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": mape,
        "medape_pct": float(np.median(
            np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None))) * 100),
        "n": int(len(y_true)),
    }
