"""API EVA-Rendimientos (T5): predicción de rendimiento agrícola por municipio.

Endpoints:
- GET  /health     → estado y archivos cargados.
- POST /predict    → rendimiento predicho (t/ha) + intervalo 90% + confianza.

La inferencia replica el espacio de 30 features de src/features.py:
lags/rolling desde history.csv (solo pasado), encodings calculados solo con
train ≤2020 en el arranque, y vector reindexado a feature_config.json.
"""
import json
import unicodedata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
TRAIN_END = 2020
SMOOTH_M = 10.0


def norm_key(s) -> str:
    """Idéntica a src/clean.py: casefold, sin acentos, espacios colapsados."""
    if not isinstance(s, str):
        return ""
    s = s.strip().casefold()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


class PredictRequest(BaseModel):
    cultivo: str
    municipio: str
    anio: int = Field(ge=2006, le=2030)
    periodo: str  # "A" | "B" | "anual"
    area_sembrada_ha: float = Field(ge=0)
    departamento: str | None = None


# --- Arranque: cargar artefactos y recalcular encodings (solo train) ---
model = lgb.Booster(model_file=str(APP_DIR / "model.txt"))
config = json.loads((APP_DIR / "feature_config.json").read_text())
history = pd.read_csv(APP_DIR / "history.csv")
cultivar_map = pd.read_csv(APP_DIR / "cultivar_map.csv")
lookup = pd.read_csv(APP_DIR / "lookup.csv")

CULT = cultivar_map.set_index("cultivo_norm").to_dict("index")
MUNI_BY_COD = lookup.set_index("cod_mun").to_dict("index")
# Nombres de municipio repetidos entre departamentos (La Unión ×4): lista por nombre
MUNI_LIST: dict[str, list] = {}
for _, row in lookup.iterrows():
    MUNI_LIST.setdefault(row["municipio_norm"], []).append(row.to_dict())

# Lags en la historia (por serie, solo pasado) para medianas de imputación
history["po"] = history["periodo_tipo"].map({"A": 0, "B": 1, "anual": 0})
history = history.sort_values(["cod_mun", "cultivo_norm", "anio", "po"])
history["grupo_norm"] = history["cultivo_norm"].map(
    {k: v["grupo_norm"] for k, v in CULT.items()})
history["cod_dep"] = history["cod_mun"].map(
    {k: v["cod_dep"] for k, v in MUNI_BY_COD.items()})
history["municipio_norm"] = history["cod_mun"].map(
    {k: v["municipio_norm"] for k, v in MUNI_BY_COD.items()})
history["r_lag1"] = history.groupby(
    ["cod_mun", "cultivo_norm"], sort=False)["rendimiento_clipped"].shift(1)

train_hist = history[history["anio"] <= TRAIN_END]
GMEAN = float(config["global_mean"])


def smoothed(col: str) -> dict:
    stats = train_hist.groupby(col)["rendimiento_clipped"].agg(["mean", "count"])
    return ((stats["count"] * stats["mean"] + SMOOTH_M * GMEAN)
            / (stats["count"] + SMOOTH_M)).to_dict()


ENC = {c: smoothed(c) for c in
       ["cultivo_norm", "municipio_norm", "cod_dep", "grupo_norm"]}
GRUPO_LAG1_MED = train_hist.groupby("grupo_norm")["r_lag1"].median().to_dict()
SERIES = {(k1, k2): g[["anio", "po", "rendimiento_clipped"]].to_numpy()
          for (k1, k2), g in history.groupby(["cod_mun", "cultivo_norm"], sort=False)}

app = FastAPI(title="EVA-Rendimientos Colombia",
              description="Predicción de rendimiento agrícola (t/ha) por municipio y cultivo.")


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "lightgbm-eva-final",
            "historial_filas": int(len(history)),
            "series": int(len(SERIES)), "cultivos": int(len(CULT)),
            "municipios": int(len(MUNI_BY_COD)), "train_end": TRAIN_END}


@app.post("/predict")
def predict(req: PredictRequest):
    cn = norm_key(req.cultivo)
    mn = norm_key(req.municipio)
    if req.periodo not in ("A", "B", "anual"):
        raise HTTPException(422, "periodo debe ser 'A', 'B' o 'anual'")
    if cn not in CULT:
        raise HTTPException(404, f"cultivo no cubierto por el modelo: {req.cultivo} "
                                 "(solo cultivos con ≥100 filas históricas)")
    cands = MUNI_LIST.get(mn, [])
    if not cands:
        raise HTTPException(404, f"municipio no encontrado: {req.municipio}")
    if len(cands) > 1:
        dept_norm = norm_key(req.departamento) if req.departamento else ""
        cands = [c for c in cands
                 if norm_key(c["departamento"]) == dept_norm] if dept_norm else []
        if len(cands) != 1:
            raise HTTPException(422, f"municipio ambiguo: {req.municipio} existe en "
                "varios departamentos; pasar departamento para desambiguar")
    muni_row = cands[0]
    grupo = CULT[cn]["grupo_norm"]
    ciclo = CULT[cn]["ciclo_norm"]
    cod_mun = int(muni_row["cod_mun"])
    cod_dep = int(muni_row["cod_dep"])
    departamento = muni_row["departamento"]

    po_req = {"A": 0, "B": 1, "anual": 0}[req.periodo]
    serie = SERIES.get((cod_mun, cn))
    hist_len = 0
    past = np.array([])
    if serie is not None:
        mask = (serie[:, 0] < req.anio) | (
            (serie[:, 0] == req.anio) & (serie[:, 1] < po_req))
        past = serie[mask][:, 2].astype(float)
        hist_len = int(len(past))
    avisos = []
    med = GRUPO_LAG1_MED.get(grupo, np.nan)
    if np.isnan(med):
        med = GMEAN
    def lag(k):  # k=1: último, k=2: penúltimo
        return float(past[-k]) if len(past) >= k else float(med)
    def roll(k):
        return float(past[-k:].mean()) if len(past) >= 1 else float(med)
    if hist_len == 0:
        avisos.append("serie sin historial previo: lags imputados con mediana "
                      "del grupo; predicción de baja confianza")
    elif hist_len < 3:
        avisos.append(f"serie corta (hist_len={hist_len}): predicción de "
                      "confianza media")

    feats = {
        "area_sembrada_log": float(np.log1p(req.area_sembrada_ha)),
        "is_B": int(req.periodo == "B"),
        "is_anual": int(req.periodo == "anual"),
        "hist_len": hist_len,
        "r_lag1": lag(1), "r_lag2": lag(2),
        "r_roll3": roll(3), "r_roll5": roll(5),
        "cultivo_norm_te": ENC["cultivo_norm"].get(cn, GMEAN),
        "municipio_norm_te": ENC["municipio_norm"].get(mn, GMEAN),
        "cod_dep_te": ENC["cod_dep"].get(cod_dep, GMEAN),
        "grupo_norm_te": ENC["grupo_norm"].get(grupo, GMEAN),
    }
    safe_grupo = grupo.replace(",", "").replace(" ", "_")
    for col in config["feature_columns"]:
        if col.startswith("grupo__"):
            feats[col] = int(safe_grupo == col[len("grupo__"):])
        elif col.startswith("ciclo__"):
            feats[col] = int(ciclo == col[len("ciclo__"):])
    X = pd.DataFrame([{c: feats.get(c, 0) for c in config["feature_columns"]}])
    pred = float(model.predict(X.to_numpy())[0])
    lo = float(max(pred + config["residual_q05"], 0.0))
    hi = float(pred + config["residual_q95"])
    confianza = "alta" if hist_len >= 5 else ("media" if hist_len >= 1 else "baja")
    return {
        "cultivo": req.cultivo, "municipio": req.municipio,
        "departamento": req.departamento or departamento,
        "anio": req.anio, "periodo": req.periodo,
        "rendimiento_predicho_t_ha": round(pred, 2),
        "intervalo_90": [round(lo, 2), round(hi, 2)],
        "hist_len": hist_len, "confianza": confianza, "avisos": avisos,
    }

class BatchRequest(BaseModel):
    cultivo: str
    anio: int = Field(ge=2006, le=2030)
    periodo: str = "A"
    area_sembrada_ha: float = Field(ge=0)
    municipios: list[str] = Field(min_length=1, max_length=200)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Predice rendimiento para múltiples municipios en una sola llamada."""
    cn = norm_key(req.cultivo)
    if cn not in CULT:
        raise HTTPException(404, f"cultivo no cubierto: {req.cultivo}")
    if req.periodo not in ("A", "B", "anual"):
        raise HTTPException(422, "periodo debe ser 'A', 'B' o 'anual'")
    results = []
    for mun_name in req.municipios:
        mn = norm_key(mun_name)
        cands = MUNI_LIST.get(mn, [])
        if not cands:
            results.append({"municipio": mun_name, "error": "no encontrado"})
            continue
        if len(cands) > 1:
            results.append({"municipio": mun_name,
                            "error": "ambiguo, pasar departamento"})
            continue
        muni_row = cands[0]
        grupo = CULT[cn]["grupo_norm"]
        ciclo = CULT[cn]["ciclo_norm"]
        cod_mun = int(muni_row["cod_mun"])
        cod_dep = int(muni_row["cod_dep"])
        po_req = {"A": 0, "B": 1, "anual": 0}[req.periodo]
        serie = SERIES.get((cod_mun, cn))
        hist_len = 0
        past = np.array([])
        if serie is not None:
            mask = (serie[:, 0] < req.anio) | (
                (serie[:, 0] == req.anio) & (serie[:, 1] < po_req))
            past = serie[mask][:, 2].astype(float)
            hist_len = int(len(past))
        med = GRUPO_LAG1_MED.get(grupo, np.nan)
        if np.isnan(med):
            med = GMEAN
        def lag(k):
            return float(past[-k]) if len(past) >= k else float(med)
        def roll(k):
            return float(past[-k:].mean()) if len(past) >= 1 else float(med)
        feats = {
            "area_sembrada_log": float(np.log1p(req.area_sembrada_ha)),
            "is_B": int(req.periodo == "B"),
            "is_anual": int(req.periodo == "anual"),
            "hist_len": hist_len,
            "r_lag1": lag(1), "r_lag2": lag(2),
            "r_roll3": roll(3), "r_roll5": roll(5),
            "cultivo_norm_te": ENC["cultivo_norm"].get(cn, GMEAN),
            "municipio_norm_te": ENC["municipio_norm"].get(mn, GMEAN),
            "cod_dep_te": ENC["cod_dep"].get(cod_dep, GMEAN),
            "grupo_norm_te": ENC["grupo_norm"].get(grupo, GMEAN),
        }
        safe_grupo = grupo.replace(",", "").replace(" ", "_")
        for col in config["feature_columns"]:
            if col.startswith("grupo__"):
                feats[col] = int(safe_grupo == col[len("grupo__"):])
            elif col.startswith("ciclo__"):
                feats[col] = int(ciclo == col[len("ciclo__"):])
        X = pd.DataFrame([{c: feats.get(c, 0) for c in config["feature_columns"]}])
        pred = float(model.predict(X.to_numpy())[0])
        lo = float(max(pred + config["residual_q05"], 0.0))
        hi = float(pred + config["residual_q95"])
        conf = "alta" if hist_len >= 5 else ("media" if hist_len >= 1 else "baja")
        results.append({
            "municipio": muni_row["municipio"],
            "departamento": muni_row["departamento"],
            "rendimiento_predicho_t_ha": round(pred, 2),
            "intervalo_90": [round(lo, 2), round(hi, 2)],
            "hist_len": hist_len,
            "confianza": conf,
        })
    return {"cultivo": req.cultivo, "anio": req.anio, "periodo": req.periodo,
            "n": len(results), "resultados": results}
