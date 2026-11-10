# T3 — Features temporales y baselines (2026-11-09)

**Estado:** hecho. Features en `src/features.py`, baselines en `src/baseline.py`,
Evidencia en `notebooks/03_baseline.ipynb` y `models/baseline_metrics.json`.
**Corrección previa:** `src/clean.py` ahora incluye `municipio_norm` en la salida.

## Features (30, sin fuga)

- Temporales por serie (municipio×cultivo, shift hacia atrás): `r_lag1`, `r_lag2`,
  `r_roll3`, `r_roll5`, `hist_len` (observaciones previas).
- Contexto: `area_sembrada_log`, `is_B`, `is_anual`, dummies de `grupo_norm`
  (15 grupos) y `ciclo_norm` (3).
- Target encoding suavizado (m=10, solo train): `cultivo_norm_te`,
  `municipio_norm_te`, `cod_dep_te`, `grupo_norm_te`.
- Imputación de lags con mediana del grupo (solo train) → media global.
- **Excluidas por fuga:** área cosechada, producción, ratio (resultado de la cosecha).

## Validación temporal

Train 2006–2020 (237.255), val 2021–2022 (47.007), test 2023–2025 (70.644).
Target encodings y scaler calculados solo en train.

## Baselines (test 2023–2025)

| Modelo | RMSE | MAE | R² | MedAPE% |
|---|---|---|---|---|
| Media histórica por cultivo | 10,02 | 5,02 | 0,576 | 34,9 |
| LinearRegression | 6,06 | 2,98 | 0,845 | 22,6 |
| **RandomForest (a superar en T4)** | **5,40** | **2,27** | **0,877** | **14,5** |

En validación (2021–2022) RF: RMSE 6,07, R² 0,831 — el test es más fácil que
la validación (mejor cobertura histórica), comportamiento esperado con lags.

## Para T4

- LightGBM + XGBoost + Optuna (misma validación + walk-forward), SHAP para importancia.
- Candidatos: modelar en log y/o por grupo si la cola pesada domina el error.
- Output: `train.py` → `models/model.joblib` + `models/metrics.json` + `models/feature_config.json`.
- API (T5) reutiliza `prepare_xy` para transformar inputs en el mismo espacio de 30 features.
