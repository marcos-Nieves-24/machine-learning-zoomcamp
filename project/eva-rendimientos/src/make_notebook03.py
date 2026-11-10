"""Construye el notebook 03_baseline.ipynb (código ejecutable, sin salidas)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (venv zoomcamp)",
    "language": "python", "name": "python3"}
md = nbf.v4.new_markdown_cell

nb.cells = [
    md("# T3 — Features temporales y baselines\n\n"
       "Validación temporal: train 2006–2020, validación 2021–2022, test 2023–2025 "
       "(`src/features.py`). Modelos: media histórica por cultivo, LinearRegression, "
       "RandomForest (`src/baseline.py`)."),
    md("## 1. Features y splits"),
    nbf.v4.new_code_cell(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path('../src')))\n"
        "from features import build_features, load_clean, prepare_xy\n\n"
        "df = load_clean()\n"
        "m = build_features(df)\n"
        "splits, gmean = prepare_xy(m)\n"
        "for name, (X, y, meta) in splits.items():\n"
        "    print(f'{name}: {len(y):,} filas, {X.shape[1]} features, "
        "media y={y.mean():.2f}')\n"
        "print('features:', list(splits['train'][0].columns))"),
    md("## 2. Métricas de los baselines (test 2023–2025)\n\n"
       "MAPE se distorsiona con rendimientos bajos: se reporta también MedAPE. "
       "Baseline a superar por el modelo final (T4): **RandomForest (RMSE 5,40; R² 0,88)**."),
    nbf.v4.new_code_cell(
        "res = json.loads(Path('../models/baseline_metrics.json').read_text())\n"
        "import pandas as pd\n"
        "rows = []\n"
        "for model in ['media_historica', 'linear', 'random_forest']:\n"
        "    for split in ['val', 'test']:\n"
        "        x = res[model][split]\n"
        "        rows.append({'modelo': model, 'corte': split,\n"
        "                     'RMSE': round(x['rmse'], 3), 'MAE': round(x['mae'], 3),\n"
        "                     'R2': round(x['r2'], 4),\n"
        "                     'MedAPE%': round(x['medape_pct'], 1),\n"
        "                     'MAPE%': round(x['mape_pct'], 1),\n"
        "                     'n': x['n']})\n"
        "print(pd.DataFrame(rows).to_string(index=False))"),
    md("## 3. Predicho vs real (test, RandomForest)"),
    nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "pred = pd.read_csv('../models/baseline_test_predictions.csv')\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
        "ax[0].scatter(pred.y_true, pred.y_pred_rf, s=2, alpha=.3)\n"
        "ax[0].plot([0, 60], [0, 60], 'k--', lw=1)\n"
        "ax[0].set_xlabel('real'); ax[0].set_ylabel('predicho (RF)')\n"
        "ax[0].set_title('Test 2023-2025'); ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 60)\n"
        "pred['res'] = pred.y_pred_rf - pred.y_true\n"
        "pred.groupby('anio')['res'].agg(['mean', 'std']).plot(ax=ax[1])\n"
        "ax[1].set_title('Residuo medio por año (estabilidad temporal)')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(pred['res'].abs().describe().to_string())"),
    md("## Conclusiones T3\n\n"
       "- La serie es predecible con historia: media histórica da R² 0,58; "
       "los lags/rolling elevan a **R² 0,85 (lineal) – 0,88 (RF)**.\n"
       "- RF es el baseline a superar en T4 (LightGBM/XGBoost + Optuna + SHAP).\n"
       "- El error crece con rendimientos altos (cola pesada) → en T4 se evalúa "
       "predicir en log o por grupo.\n"
       "- Pregunta abierta para T4: ¿el encoding por municipio captura "
       "edafoclimática o sobreajusta municipios con poca historia? "
       "Se regula con suavizado (m=10) y se mide con walk-forward."),
]

nbf.write(nb, "/home/biohacker24/proyectos/second-brain/eva-rendimientos/notebooks/03_baseline.ipynb")
print("Notebook creado.")
