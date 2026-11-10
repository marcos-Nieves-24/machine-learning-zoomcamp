"""Limpieza T2: normalización, consistencia, outliers y cultivo raro.

Lee los CSV crudos de data/eva_raw/, aplica las decisiones del reporte T1
(y documentadas con base en la metodología EVA de UPRA) y guarda:
- data/eva_clean/eva_clean.csv  (57 columnas limpias, ordenadas)
- data/eva_clean/cleaning_log.json (conteo de cada paso)

La función clean_eva() es reutilizable desde el notebook 02 y desde features.py.
"""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ROWS_PER_CULTIVO = 100


def norm_key(s) -> str:
    """Clave normalizada para agrupar variantes: casefold, sin acentos, sin espacios extra."""
    if not isinstance(s, str):
        return ""
    s = s.strip().casefold()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def clean_eva() -> tuple[pd.DataFrame, dict]:
    base = Path(__file__).resolve().parent.parent / "data"
    # Archivos crudos ignorados por .gitignore; si no existen, indicar re-ejecutar ingest.
    paths = [base / "eva_raw" / "eva_2006_2018.csv",
             base / "eva_raw" / "eva_2019_2025.csv"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan crudos: {missing}. Re-ejecutar: python3 src/ingest.py")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    log: dict = {"raw_rows": int(len(df))}

    # 1. Numéricos
    for c in ["anio", "area_sembrada_ha", "area_cosechada_ha",
              "produccion_t", "rendimiento_t_ha"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log["numeric_nulls_after_coerce"] = {
        c: int(df[c].isna().sum()) for c in
        ["anio", "area_sembrada_ha", "area_cosechada_ha",
         "produccion_t", "rendimiento_t_ha"]}

    # 2. Normalización de texto (MAIZ vs Maíz duplican series)
    for c in ["cultivo", "grupo_cultivo", "subgrupo", "departamento",
              "municipio", "estado_fisico"]:
        df[c] = df[c].astype(str).str.strip()
    df["cultivo_norm"] = df["cultivo"].map(norm_key)
    df["grupo_norm"] = df["grupo_cultivo"].map(norm_key)
    df["municipio_norm"] = df["municipio"].map(norm_key)
    df["ciclo_norm"] = df["ciclo_cultivo"].astype(str).str.strip().str.casefold()
    df["ciclo_norm"] = df["ciclo_norm"].replace(
        {"transitorio": "transitorio", "permanente": "permanente",
         "anual": "anual"})

    # 3. Periodo: anual (2024) vs A/B (2024A/2024B)
    periodo = df["periodo"].astype(str).str.strip()
    df["periodo_tipo"] = np.where(
        periodo.str.endswith("A"), "A",
        np.where(periodo.str.endswith("B"), "B", "anual"))
    df["periodo_num"] = pd.to_numeric(
        periodo.str.replace(r"[AB]$", "", regex=True), errors="coerce")

    # 4. Consistencia técnica: producción ≈ rendimiento × área_cosechada (UPRA la define así)
    expected = df["rendimiento_t_ha"] * df["area_cosechada_ha"]
    prod = df["produccion_t"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_err"] = (prod - expected).abs() / prod.where(prod > 0)
    df["inconsistente"] = (
        (prod > 0) & (df["area_cosechada_ha"] > 0)
        & df["rendimiento_t_ha"].notna() & (df["rel_err"] > 0.20))
    log["inconsistentes_gt20pct"] = int(df["inconsistente"].sum())

    # 5. Target: sin rendimiento o sin área cosechada no es modelable (rendimiento indefinido)
    df["target_modelable"] = (
        df["rendimiento_t_ha"].notna() & (df["area_cosechada_ha"] > 0))
    log["target_no_modelable"] = int((~df["target_modelable"]).sum())
    log["target_modelable"] = int(df["target_modelable"].sum())

    # 6. Winsorización de rendimiento (cota 99.5%) — conserva ceros (cultivos perdidos)
    valid = df.loc[df["target_modelable"], "rendimiento_t_ha"]
    p995 = float(valid.quantile(0.995))
    p99 = float(valid.quantile(0.99))
    df["rendimiento_clipped"] = df["rendimiento_t_ha"].clip(0, p995)
    log["rendimiento_p99"] = p99
    log["rendimiento_p995_cota"] = p995
    log["rows_capped"] = int((df["rendimiento_t_ha"] > p995).sum())

    # 7. Cultivo raro (<100 filas) — agrupar por grupo en T3/T4
    counts = df["cultivo_norm"].value_counts()
    df["cultivo_raro"] = df["cultivo_norm"].map(counts) < MIN_ROWS_PER_CULTIVO
    log["cultivos_raros"] = int((counts < MIN_ROWS_PER_CULTIVO).sum())
    log["rows_in_rare_cultivos"] = int(df["cultivo_raro"].sum())
    log["distinct_cultivos_norm"] = int(counts.size)

    # 8. Razón de cosecha (eficiencia)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio_cosecha"] = np.where(
            df["area_sembrada_ha"] > 0,
            df["area_cosechada_ha"] / df["area_sembrada_ha"], np.nan)
    log["ratio_gt_1pct"] = float((df["ratio_cosecha"] > 1).mean() * 100)

    # 9. Orden y selección de columnas limpias
    keep = ["cod_dep", "departamento", "cod_mun", "municipio",
            "municipio_norm", "grupo_cultivo", "grupo_norm", "subgrupo", "cultivo",
            "cultivo_norm", "desagregacion", "anio", "periodo",
            "periodo_tipo", "periodo_num", "area_sembrada_ha",
            "area_cosechada_ha", "produccion_t", "rendimiento_t_ha",
            "rendimiento_clipped", "estado_fisico", "nombre_cientifico",
            "ciclo_norm", "cod_cultivo", "fuente", "rel_err",
            "inconsistente", "target_modelable", "cultivo_raro",
            "ratio_cosecha"]
    df = df[keep].sort_values(["cod_mun", "cultivo_norm", "anio", "periodo"])
    df.to_csv(base / "eva_clean" / "eva_clean.csv", index=False)
    (base / "eva_clean" / "cleaning_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    return df, log


def main() -> None:
    (Path(__file__).resolve().parent.parent / "data"
     / "eva_clean").mkdir(parents=True, exist_ok=True)
    df, log = clean_eva()
    print(json.dumps(log, indent=2))
    print(f"Filas limpias: {len(df)} "
          f"(modelables: {log['target_modelable']})")


if __name__ == "__main__":
    main()
