"""Construye el notebook 02_cleaning_eda.ipynb (código ejecutable, sin salidas)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (venv zoomcamp)",
    "language": "python", "name": "python3"}

md = nbf.v4.new_markdown_cell

nb.cells = [
    md("# T2 — Limpieza y EDA EVA-Rendimientos\n\n"
       "Limpieza reproducible con `src/clean.py` (decisiones y evidencia en "
       "`T2-limpieza-eda.md`). Fuente: EVA Colombia 2006–2025 "
       "(`2pnw-mmge` + `uejq-wxrr`, datos.gov.co)."),
    md("## 1. Carga y limpieza"),
    nbf.v4.new_code_cell(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path('../src')))\n"
        "from clean import clean_eva\n\n"
        "df, log = clean_eva()\n"
        "print(json.dumps(log, indent=2))\n"
        "print('Filas limpias:', len(df), '| modelables:', log['target_modelable'])"),
    md("## 2. Distribución del target (rendimiento t/ha)\n\n"
       "Ceros = cultivo perdido (definido en UPRA: área cosechada 0 → "
       "rendimiento 0). p99 = 85 t/ha; cota de winsorización (p99.5) medida en celda 1."),
    nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "r = df.loc[df.target_modelable, 'rendimiento_t_ha']\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].hist(r, bins=50)\n"
        "ax[0].set_title('Rendimiento t/ha (modelable)')\n"
        "ax[1].hist(r[r < 30], bins=50)\n"
        "ax[1].set_title('Rendimiento t/ha (<30)')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(r.describe(percentiles=[.01,.5,.9,.99,.995]).to_string())\n"
        "print('ceros:', round((r == 0).mean()*100, 2), '%')"),
    md("## 3. Cobertura temporal y cultivos principales\n\n"
       "Constante: filtrar cultivos con <100 filas en T3/T4 (119 cultivos, 3.148 filas)."),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df['anio'].value_counts().sort_index().plot(ax=ax[0])\n"
        "ax[0].set_title('Filas por año'); ax[0].set_ylabel('filas')\n"
        "df['cultivo_norm'].value_counts().head(15).plot.barh(ax=ax[1])\n"
        "ax[1].set_title('Top 15 cultivos (normalizados)'); ax[1].invert_xaxis()\n"
        "plt.tight_layout(); plt.show()\n"
        "print('cultivos normalizados:', df.cultivo_norm.nunique(), "
        "'| municipios:', df.cod_mun.nunique())"),
    md("## 4. Rendimiento por grupo de cultivo y por ciclo\n\n"
       "La variabilidad entre grupos justifica target encoding por grupo/cultivo "
       "(decisión T3/T4)."),
    nbf.v4.new_code_cell(
        "g = (df[df.target_modelable]\n"
        "     .groupby('grupo_norm')['rendimiento_clipped']\n"
        "     .median().sort_values())\n"
        "print(g.to_string())\n"
        "print()\n"
        "print(df.groupby('ciclo_norm')['rendimiento_clipped']\n"
        "      .agg(['count', 'median', 'mean']).to_string())"),
    md("## 5. Calidad: consistencia técnica y razón de cosecha\n\n"
       "UPRA define producción = rendimiento × área cosechada. Filas con error "
       "relativo >20% se marcan `inconsistente` (3.631 filas, ~1%) y se revisan "
       "en T2 antes de modelar. Ratio cosechada/sembrada >1 en 2,75% (inconsistencia "
       "de reporte municipal)."),
    nbf.v4.new_code_cell(
        "print('inconsistentes:', int(df.inconsistente.sum()), "
        "'(', round(df.inconsistente.mean()*100, 2), '%)')\n"
        "print('ratio>1:', round((df.ratio_cosecha > 1).mean()*100, 2), '%')\n"
        "print('target no modelable:', int((~df.target_modelable).sum()))\n"
        "print(df.groupby('fuente').size().to_string())"),
    md("## Conclusiones T2\n\n"
       "- 361.322 filas modelables (96,9% del total); 11.478 filas no modelables "
       "(área cosechada 0 o target nulo).\n"
       "- Normalización de texto fusionó 388 cultivos crudos → 248 normalizados "
       "(MAIZ/Maíz).\n"
       "- Winsorización en p99.5 (113,3 t/ha) afecta 1.845 filas; ceros conservados.\n"
       "- 119 cultivos raros (<100 filas) con solo 3.148 filas → se filtran o "
       "agrupan por `grupo_norm` en T3.\n"
       "- Siguiente paso (T3): features temporales (lags, rolling) + baselines "
       "con validación temporal."),
]

nbf.write(nb, "/home/biohacker24/proyectos/second-brain/eva-rendimientos/notebooks/02_cleaning_eda.ipynb")
print("Notebook creado.")
