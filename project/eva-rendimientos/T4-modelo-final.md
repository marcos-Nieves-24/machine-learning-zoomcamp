# T4 — Modelo final (2026-11-09)

**Estado:** hecho. Entrenado con `src/train.py`, evidencia en `notebooks/04_model.ipynb`.
**Artefactos:** `models/model.joblib` (8,6MB), `models/metrics.json`,
`models/feature_config.json`, `models/test_predictions.csv`.

## Optimización (Optuna, 20 trials, train ≤2020 → val 2021–2022)

Mejor val RMSE 5,843 con: n_estimators 580, learning_rate 0,0116, num_leaves 177,
min_child_samples 18, subsample 0,918, colsample 0,692, reg_lambda 11,07.

## Resultados principales

| Modelo | Test RMSE | Test MAE | Test R² | Test MedAPE |
|---|---|---|---|---|
| RandomForest (T3) | 5,40 | 2,27 | 0,877 | 14,5% |
| **LightGBM final** | **4,96** | **1,85** | **0,896** | **10,3%** |

Val (2021–2022): RMSE 5,84, R² 0,843.

## Walk-forward (ventanas expansivas)

| Vuelta | LightGBM R² | XGBoost R² |
|---|---|---|
| train ≤2019 → test 2020 | 0,868 | 0,841 |
| train ≤2020 → test 2021 | 0,887 | 0,864 |
| train ≤2021 → test 2022 | 0,897 | 0,876 |

Estable y creciente (mejor cobertura histórica en vueltas posteriores) → sin drift
alarmante. XGBoost pierde en todas las vueltas → fuera del deploy (comparación documentada).

## SHAP (modelo final sobre test)

Top: `r_roll3` (3,00), `r_lag1` (2,95), `cultivo_norm_te` (1,38), `r_roll5` (0,79),
`r_lag2` (0,72), `municipio_norm_te` (0,33), `area_sembrada_log` (0,27), `hist_len` (0,17).
La historia manda: el rendimiento se autocorrelaciona. Municipios sin historia
(`hist_len` = 0) → predicción degradada (documentar en README).

## Reentrenamiento final

`model.joblib` es el modelo con los best params entrenado con **todo 2006–2025**
(walk-forward elige el recipe; el recipe entrena el modelo final). Residuo medio
en test: −0,27 (ligera subestimación sistemática, aceptable).

## Para T5

- API FastAPI `/predict`: input (cultivo, municipio, depto, área sembrada, periodo decide
  lags desde la DB o desde payload) → reutilizar encodings de `feature_config.json`.
  Aclaración: los lags requieren historia de la serie; la API necesita una tabla
  histórica (SQLite) o que el cliente pase los lags. Ver diseño en T5.
