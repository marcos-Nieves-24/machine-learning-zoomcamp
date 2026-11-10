"""Construye el notebook 04_model.ipynb (código ejecutable, sin salidas)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (venv zoomcamp)",
    "language": "python", "name": "python3"}
md = nbf.v4.new_markdown_cell

nb.cells = [
    md("# T4 — Modelo final: LightGBM + Optuna + walk-forward + SHAP\n\n"
       "Entrenado con `src/train.py`. Comparación final vs baselines de T3. "
       "Artefactos en `models/` (`model.joblib`, `metrics.json`, `feature_config.json`)."),
    md("## 1. Resultados: baselines vs final (test 2023–2025)"),
    nbf.v4.new_code_cell(
        "import json\nimport pandas as pd\n"
        "from pathlib import Path\n\n"
        "base = json.loads(Path('../models/baseline_metrics.json').read_text())\n"
        "final = json.loads(Path('../models/metrics.json').read_text())\n"
        "rows = [\n"
        "    {'modelo': 'media historica', 'RMSE': round(base['media_historica']['test']['rmse'], 2),\n"
        "     'MAE': round(base['media_historica']['test']['mae'], 2), 'R2': round(base['media_historica']['test']['r2'], 4)},\n"
        "    {'modelo': 'linear', 'RMSE': round(base['linear']['test']['rmse'], 2),\n"
        "     'MAE': round(base['linear']['test']['mae'], 2), 'R2': round(base['linear']['test']['r2'], 4)},\n"
        "    {'modelo': 'random forest', 'RMSE': round(base['random_forest']['test']['rmse'], 2),\n"
        "     'MAE': round(base['random_forest']['test']['mae'], 2), 'R2': round(base['random_forest']['test']['r2'], 4)},\n"
        "    {'modelo': 'lightgbm (final)', 'RMSE': round(final['test']['rmse'], 2),\n"
        "     'MAE': round(final['test']['mae'], 2), 'R2': round(final['test']['r2'], 4)},\n"
        "]\n"
        "print(pd.DataFrame(rows).to_string(index=False))\n"
        "print()\n"
        "print('Best params Optuna (val RMSE %.3f):' % final['optuna_best_val_rmse'])\n"
        "print(json.dumps(final['optuna_best_params'], indent=2))"),
    md("## 2. Walk-forward: estabilidad año a año (LightGBM vs XGBoost)\n\n"
       "Ventanas expansivas: train ≤2019→test 2020; ≤2020→2021; ≤2021→2022. "
       "R² estable y creciente en LightGBM (0,868→0,887→0,897): sin drift alarmante."),
    nbf.v4.new_code_cell(
        "wf = final['walk_forward']\n"
        "wrows = []\n"
        "for model in ['lightgbm', 'xgboost']:\n"
        "    for fold in wf[model]:\n"
        "        fold_name, m = next(iter(fold.items()))\n"
        "        wrows.append({'modelo': model, 'vuelta': fold_name,\n"
        "                      'RMSE': round(m['rmse'], 2), 'R2': round(m['r2'], 4),\n"
        "                      'n': m['n']})\n"
        "print(pd.DataFrame(wrows).to_string(index=False))"),
    md("## 3. Predicho vs real + residuos por año (test)"),
    nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "pred = pd.read_csv('../models/test_predictions.csv')\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
        "ax[0].scatter(pred.y_true, pred.y_pred, s=2, alpha=.3)\n"
        "ax[0].plot([0, 60], [0, 60], 'k--', lw=1)\n"
        "ax[0].set_xlabel('real'); ax[0].set_ylabel('predicho (final)')\n"
        "ax[0].set_title('Test 2023-2025'); ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 60)\n"
        "pred['res'] = pred.y_pred - pred.y_true\n"
        "pred.groupby('anio')['res'].agg(['mean', 'std']).plot(ax=ax[1])\n"
        "ax[1].set_title('Residuo por año')\n"
        "plt.tight_layout(); plt.show()\n"
        "print('error abs medio:', pred['res'].abs().mean().round(2), "
        " '| residuo medio:', pred['res'].mean().round(3))"),
    md("## 4. Importancia SHAP (modelo final sobre test)\n\n"
       "Dominan las features históricas (r_roll3, r_lag1, encoding de cultivo): "
       "el rendimiento se autocorrelaciona. `area_sembrada_log` aporta señal "
       "marginal; `hist_len` (profundidad de la serie) también."),
    nbf.v4.new_code_cell(
        "import joblib, shap, sys\n"
        "sys.path.insert(0, str(Path('../src')))\n"
        "from features import build_features, load_clean, prepare_xy\n\n"
        "df = load_clean()\n"
        "m = build_features(df)\n"
        "splits, _ = prepare_xy(m, train_end=2020, val_end=2022)\n"
        "Xte = splits['test'][0]\n"
        "model = joblib.load('../models/model.joblib')\n"
        "explainer = shap.TreeExplainer(model)\n"
        "sv = explainer.shap_values(Xte.head(5000))\n"
        "shap.summary_plot(sv, Xte.head(5000), show=False, max_display=12)\n"
        "plt.tight_layout(); plt.show()"),
    md("## Conclusiones T4\n\n"
       "- LightGBM final: **test RMSE 4,96 / MAE 1,85 / R² 0,896 / MedAPE 10,3%** "
       "(mejora RF 5,40/0,877).\n"
       "- Walk-forward estable (R² 0,87–0,90 en 3 años): el modelo no depende de un año con suerte.\n"
       "- XGBoost gana en nada: queda fuera del deploy (se documenta la comparación).\n"
       "- SHAP: la historia manda (lags/rolling + encoding). Si el modelo se usa "
       "para municipios nuevos (sin historia), rendimiento degradado → documentado en README.\n"
       "- Siguiente paso (T5): API FastAPI + Docker reutilizando `feature_config.json` "
       "y `model.joblib`; endpoint `/predict` con un intervalo de predicción."),
]

nbf.write(nb, "/home/biohacker24/proyectos/second-brain/eva-rendimientos/notebooks/04_model.ipynb")
print("Notebook creado.")
