"""Verificación T1: estructura, nulos, outliers, distribuciones y cobertura.

Lee los CSV crudos de data/eva_raw/, los unifica y genera data/verification_report.json
con evidencia para decidir la limpieza (T2). No modifica los crudos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUM_COLS = ["anio", "area_sembrada_ha", "area_cosechada_ha",
            "produccion_t", "rendimiento_t_ha"]


def describe_numeric(s: pd.Series) -> dict:
    return {
        "count": int(s.count()),
        "nulls": int(s.isna().sum()),
        "min": float(s.min()) if s.count() else None,
        "max": float(s.max()) if s.count() else None,
        "mean": float(s.mean()) if s.count() else None,
        "median": float(s.median()) if s.count() else None,
        "p1": float(s.quantile(0.01)) if s.count() else None,
        "p99": float(s.quantile(0.99)) if s.count() else None,
        "zeros": int((s == 0).sum()),
        "negatives": int((s < 0).sum()),
    }


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "data"
    hist = pd.read_csv(base / "eva_raw" / "eva_2006_2018.csv")
    recent = pd.read_csv(base / "eva_raw" / "eva_2019_2025.csv")
    df = pd.concat([hist, recent], ignore_index=True)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    r = df["rendimiento_t_ha"]
    valid_r = r.dropna()
    report = {
        "total_rows": int(len(df)),
        "hist_rows": int(len(hist)),
        "recent_rows": int(len(recent)),
        "columns": list(df.columns),
        "column_count": int(df.shape[1]),
        "years_min": int(df["anio"].min()),
        "years_max": int(df["anio"].max()),
        "n_municipios": int(df["cod_mun"].nunique()),
        "n_departamentos": int(df["cod_dep"].nunique()),
        "n_cultivos": int(df["cultivo"].nunique()),
        "n_grupos": int(df["grupo_cultivo"].nunique()),
        "nulls_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_summary": {c: describe_numeric(df[c]) for c in NUM_COLS},
        "target_non_null": int(valid_r.count()),
        "target_null_pct": round(100 * float(r.isna().mean()), 3),
        "target_zero_pct": round(100 * float((valid_r == 0).mean()), 3),
        # outliers extremos de rendimiento: pct 1/99 y fuera de [p0.001, p0.999]
        "target_p01": float(valid_r.quantile(0.01)),
        "target_p99": float(valid_r.quantile(0.99)),
        "target_extreme_high_gt_p999": int((valid_r > valid_r.quantile(0.999)).sum()),
        "periodos": df["periodo"].value_counts().head(20).to_dict(),
        "ciclos": df["ciclo_cultivo"].value_counts().head(10).to_dict(),
        "top_cultivos": df["cultivo"].value_counts().head(15).to_dict(),
        "cultivos_menor_100_filas": int((df["cultivo"].value_counts() < 100).sum()),
        "rows_per_year": df["anio"].value_counts().sort_index().to_dict(),
        # consistencia: produccion ~= rendimiento * area_cosechada
        "produccion_vs_rend_x_area_max_rel_err": float(
            ((df["produccion_t"] - r * df["area_cosechada_ha"]).abs()
             / df["produccion_t"].replace(0, np.nan)).max()),
    }
    out = base / "verification_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Reporte -> {out}")
    # Resumen legible
    print(f"Filas totales: {report['total_rows']} "
          f"(hist {report['hist_rows']} + reciente {report['recent_rows']})")
    print(f"Años: {report['years_min']}-{report['years_max']} | "
          f"Municipios: {report['n_municipios']} | "
          f"Cultivos: {report['n_cultivos']} | Grupos: {report['n_grupos']}")
    print(f"Target válido: {report['target_non_null']} "
          f"(nulos {report['target_null_pct']}%, ceros {report['target_zero_pct']}%)")
    print(f"Rendimiento: min={report['numeric_summary']['rendimiento_t_ha']['min']} "
          f"p1={report['target_p01']:.2f} p99={report['target_p99']:.2f} "
          f"max={report['numeric_summary']['rendimiento_t_ha']['max']}")
    print(f"Cultivos con <100 filas: {report['cultivos_menor_100_filas']}")


if __name__ == "__main__":
    main()
