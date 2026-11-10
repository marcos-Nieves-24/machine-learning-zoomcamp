"""Exporta artefactos compactos de inferencia para api/ (T5).

Lee data/eva_clean/eva_clean.csv y escribe en api/:
- history.csv: serie (cod_mun, cultivo_norm, anio, periodo_tipo, rendimiento_clipped)
  solo filas modelables, consistentes, no raras (= universo de entrenamiento).
- cultivar_map.csv: cultivo_norm -> grupo_norm, ciclo_norm (248).
- lookup.csv: cod_mun -> municipio_norm, departamento, cod_dep.

La API recalcula encodings (solo train <=2020) y medianas de lags en el arranque
desde estos archivos: no se versionan mapas duplicados.
"""
from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base / "data" / "eva_clean" / "eva_clean.csv")
    m = df[df["target_modelable"] & ~df["inconsistente"]
          & ~df["cultivo_raro"]].copy()
    api = base / "api"
    api.mkdir(parents=True, exist_ok=True)
    hist = m[["cod_mun", "cultivo_norm", "anio", "periodo_tipo",
              "rendimiento_clipped"]].sort_values(["cod_mun", "cultivo_norm", "anio"])
    hist.to_csv(api / "history.csv", index=False)
    cmap = (m.groupby("cultivo_norm")[["cultivo", "grupo_norm", "ciclo_norm"]]
            .first().reset_index())
    cmap.to_csv(api / "cultivar_map.csv", index=False)
    lk = (m.groupby("cod_mun")[["municipio", "municipio_norm", "departamento", "cod_dep"]]
          .first().reset_index())
    lk.to_csv(api / "lookup.csv", index=False)
    print(f"history: {len(hist):,} filas -> {api / 'history.csv'} "
          f"({(api / 'history.csv').stat().st_size / 1e6:.1f} MB)")
    print(f"cultivar_map: {len(cmap)} cultivos")
    print(f"lookup: {len(lk)} municipios")


if __name__ == "__main__":
    main()
