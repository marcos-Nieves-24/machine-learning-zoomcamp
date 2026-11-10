"""Ingesta EVA Colombia desde Socrata (datos.gov.co).

Descarga EVA 2006-2018 (2pnw-mmge) y EVA 2019-2025 (uejq-wxrr) con paginación,
unifica nombres de columna y guarda CSV crudos en data/eva_raw/.
Mapeo verificado contra /api/views/{id} el 2026-11-09.
"""
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/{id}.csv"
PAGE = 50000

# Nombre en Socrata -> nombre unificado
COLUMN_MAP = {
    # histórico (2pnw-mmge)
    "c_d_dep": "cod_dep",
    "departamento": "departamento",
    "c_d_mun": "cod_mun",
    "municipio": "municipio",
    "grupo_de_cultivo": "grupo_cultivo",
    "subgrupo_de_cultivo": "subgrupo",
    "cultivo": "cultivo",
    "desagregaci_n_regional_y": "desagregacion",
    "a_o": "anio",
    "periodo": "periodo",
    "rea_sembrada_ha": "area_sembrada_ha",
    "rea_cosechada_ha": "area_cosechada_ha",
    "producci_n_t": "produccion_t",
    "rendimiento_t_ha": "rendimiento_t_ha",
    "estado_fisico_produccion": "estado_fisico",
    "nombre_cientifico": "nombre_cientifico",
    "ciclo_de_cultivo": "ciclo_cultivo",
    # reciente (uejq-wxrr) — sin sufijos, con codigo del cultivo extra
    "c_digo_dane_departamento": "cod_dep",
    "c_digo_dane_municipio": "cod_mun",
    "grupo_cultivo": "grupo_cultivo",
    "desagregaci_n_cultivo": "desagregacion",
    "rea_sembrada": "area_sembrada_ha",
    "rea_cosechada": "area_cosechada_ha",
    "producci_n": "produccion_t",
    "rendimiento": "rendimiento_t_ha",
    "ciclo_del_cultivo": "ciclo_cultivo",
    "estado_f_sico_del_cultivo": "estado_fisico",
    "c_digo_del_cultivo": "cod_cultivo",
    "nombre_cient_fico_del_cultivo": "nombre_cientifico",
    "subgrupo": "subgrupo",
}

UNIFIED_COLUMNS = [
    "cod_dep", "departamento", "cod_mun", "municipio",
    "grupo_cultivo", "subgrupo", "cultivo", "desagregacion",
    "anio", "periodo", "area_sembrada_ha", "area_cosechada_ha",
    "produccion_t", "rendimiento_t_ha", "estado_fisico",
    "nombre_cientifico", "ciclo_cultivo", "cod_cultivo",
    "fuente",
]


def download_dataset(dataset_id: str, out_path: Path) -> int:
    """Descarga paginada un dataset Socrata -> CSV con columnas unificadas."""
    total = 0
    offset = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        while True:
            qs = urllib.parse.urlencode({"$limit": PAGE, "$offset": offset})
            url = f"{BASE.format(id=dataset_id)}?{qs}"
            with urllib.request.urlopen(url, timeout=120) as resp:
                text = resp.read().decode("utf-8")
            lines = text.strip().splitlines()
            if len(lines) <= 1:
                break
            reader = csv.DictReader(lines)
            rows = list(reader)
            for r in rows:
                row = {COLUMN_MAP.get(k, k): v for k, v in r.items()
                       if k in COLUMN_MAP}
                row["fuente"] = dataset_id
                for c in UNIFIED_COLUMNS:
                    row.setdefault(c, "")
                writer.writerow({c: row[c] for c in UNIFIED_COLUMNS})
            total += len(rows)
            offset += len(rows)
            print(f"  {dataset_id}: {total} filas...")
            if len(rows) < PAGE:
                break
    return total


def main() -> None:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "eva_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ds_id, fname in [("2pnw-mmge", "eva_2006_2018.csv"),
                         ("uejq-wxrr", "eva_2019_2025.csv")]:
        out = raw_dir / fname
        print(f"Descargando {ds_id} -> {out}")
        n = download_dataset(ds_id, out)
        manifest[fname] = n
        print(f"Listo: {n} filas en {out}")
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
