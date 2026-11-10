# EVA-Rendimientos: predicción de rendimiento agrícola por municipio en Colombia

Modelo de machine learning que predice el rendimiento agrícola (t/ha) por
municipio, cultivo y año, a partir de 20 años de Evaluaciones Agropecuarias
Municipales (EVA) de Colombia. Desplegado como API FastAPI + Docker.

**Midterm Project — ML Zoomcamp 2026 (DataTalks.Club)**

## Resultado

| Métrica (test 2023–2025) | Valor |
|---|---|
| RMSE | 4,96 t/ha |
| MAE | 1,85 t/ha |
| R² | 0,896 |
| MedAPE | 10,3 % |

Baseline a superar (RandomForest, T3): RMSE 5,40 / R² 0,877.
Mejora relativa: −8,1 % en RMSE, +2,2 % en R².

Walk-forward (ventanas expansivas, 3 años): R² 0,868 → 0,887 → 0,897 (estable,
sin drift). XGBoost queda fuera del deploy (R² 0,841–0,876 en las mismas vueltas).

## Dataset

**EVA (Evaluaciones Agropecuarias Municipales)** — UPRA / MADR, datos abiertos
Colombia (`datos.gov.co`).

| Bloque | ID Socrata | Filas | Cobertura |
|---|---|---|---|
| EVA 2006–2018 | `2pnw-mmge` | 206.068 | Histórico |
| EVA 2019–2025 | `uejq-wxrr` | 166.732 | Reciente |
| **Total** | | **372.800** | **2006–2025, 1.109 municipios, 32 departamentos** |

Columnas clave: departamento, municipio, grupo/subgrupo/cultivo, año, periodo
(A/B/anual), área sembrada (ha), área cosechada (ha), producción (t),
rendimiento (t/ha), ciclo (permanente/transitorio).

Licencia: datos abiertos Colombia (sin restricción comercial). Reproducible con
`src/ingest.py` (descarga paginada de Socrata).

## Arquitectura

```
datos.gov.co (Socrata)
        │
        ▼
  src/ingest.py  ──▶  data/eva_raw/*.csv
        │
        ▼
  src/clean.py   ──▶  data/eva_clean/eva_clean.csv  (372.800 filas)
        │
        ▼
  src/features.py ──▶  30 features (lags, rolling, target encoding, dummies)
        │
        ▼
  src/train.py   ──▶  models/model.joblib (LightGBM, 580 árboles)
        │
        ▼
  api/app.py     ──▶  FastAPI /predict (t/ha + intervalo 90%)
        │
        ▼
  api/Dockerfile ──▶  imagen reproducible
```

## Features (30, sin fuga)

- **Temporales** (por serie municipio×cultivo, solo pasado): `r_lag1`,
  `r_lag2`, `r_roll3`, `r_roll5`, `hist_len`.
- **Contexto:** `area_sembrada_log`, `is_B`, `is_anual`, dummies de grupo
  (15) y ciclo (3).
- **Target encoding** suavizado (m=10, calculado solo en train): cultivo,
  municipio, departamento, grupo.
- **Excluidas por fuga:** área cosechada, producción, ratio (son resultado de
  la cosecha; a la hora de sembrar no se conocen).

## Validación

- Split temporal: train 2006–2020 (237.255), val 2021–2022 (47.007),
  test 2023–2025 (70.644).
- Walk-forward expansivo (3 vueltas): train ≤2019→2020, ≤2020→2021,
  ≤2021→2022. R² estable 0,87–0,90.
- Optuna (20 trials) sobre el corte principal; hiperparámetros en
  `models/feature_config.json`.

## SHAP (importancia)

Mandan `r_roll3` (3,00), `r_lag1` (2,95), `cultivo_norm_te` (1,38):
el rendimiento se autocorrelaciona. Municipios sin historia (`hist_len` = 0)
reciben predicción de baja confianza (lags imputados con mediana del grupo).

## Setup

### Local (sin Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # dependencias del proyecto
pip install -r api/requirements.txt
python src/ingest.py                # descarga EVA (57 MB)
python src/clean.py                 # limpieza
python src/baseline.py              # baselines T3 (opcional)
python src/train.py                 # entrena el modelo final
python src/build_history.py         # exporta history.csv para la API
cp models/model.joblib models/feature_config.json api/
uvicorn api.app:app --app-dir . --port 8000
```

### Docker

```bash
cd api
docker build -t eva-rendimientos .
docker run -p 8000:8000 eva-rendimientos
```

### Endpoints

- `GET /health` → estado y archivos cargados.
- `POST /predict` → rendimiento predicho (t/ha) + intervalo 90% + confianza.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cultivo":"Maíz","municipio":"Campoalegre","anio":2026,"periodo":"A","area_sembrada_ha":200,"departamento":"Huila"}'
```

Respuesta:

```json
{
  "cultivo": "Maíz",
  "municipio": "Campoalegre",
  "departamento": "Huila",
  "anio": 2026,
  "periodo": "A",
  "rendimiento_predicho_t_ha": 4.06,
  "intervalo_90": [0.0, 7.55],
  "hist_len": 96,
  "confianza": "alta",
  "avisos": []
}
```

## Estructura del repo

```
eva-rendimientos/
├── src/
│   ├── ingest.py           # descarga Socrata (paginada)
│   ├── verify.py           # verificación de datos (T1)
│   ├── clean.py            # limpieza + normalización (T2)
│   ├── features.py         # feature engineering sin fuga (T3)
│   ├── baseline.py         # baselines (media, lineal, RF)
│   ├── train.py            # Optuna + walk-forward + SHAP (T4)
│   └── build_history.py    # exporta history.csv para la API (T5)
├── api/
│   ├── app.py              # FastAPI
│   ├── requirements.txt
│   └── Dockerfile
├── notebooks/
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_model.ipynb
├── tests/
│   └── smoke_test.py       # 5 casos (health, alta/baja confianza, 404, 422)
├── models/
│   ├── model.joblib
│   ├── metrics.json
│   └── feature_config.json
├── T1-ingesta-verificacion.md
├── T2-limpieza-eda.md
├── T3-baseline.md
├── T4-modelo-final.md
└── README.md
```

## Decisiones de diseño

- **Validación temporal, no aleatoria ni KFold:** con series temporales, un split aleatorio
  contamina lags y target encoding con el futuro; KFold baraja filas y rompe el orden.
  Walk-forward mide el escenario real de deploy ("¿qué tan bien predigo la próxima cosecha?").
- **Sin fuga:** área cosechada, producción y ratio se excluyen de las features porque
  son resultado de la cosecha.
- **Target encoding solo con train:** evita contaminar validación/test con el futuro.
- **MedAPE como métrica principal** (no MAPE): el MAPE se distorsiona con rendimientos
  bajos (0,5 t/ha); la mediana es robusta.
- **Intervalo 90%** basado en residuos del test (q05/q95), no en supuestos paramétricos.

## Fuentes

- EVA 2006–2018: https://www.datos.gov.co/Agricultura-y-Desarrollo-Rural/Evaluaciones-Agropecuarias-Municipales-EVA/2pnw-mmge
- EVA 2019–2025: https://www.datos.gov.co/Agricultura-y-Desarrollo-Rural/Evaluaciones-Agropecuarias-Municipales-EVA-2019-20/uejq-wxrr
- Metodología EVA (UPRA): https://upra.gov.co/es-co/eva
- Paper AClimate Colombia: https://doi.org/10.1016/j.compag.2020.105486

## Dashboard (Streamlit)

Dashboard interactivo para explorar predicciones y datos históricos.

### Ejecutar

```bash
cd eva-rendimientos
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Abre http://localhost:8501

### Tabs

1. **Predicción:** seleccioná cultivo + municipio + año + área sembrada → rendimiento predicho + intervalo 90% + confianza.
2. **Histórico:** línea de tiempo del rendimiento real del municipio×cultivo + predicción superpuesta.
3. **Mapa nacional:** coroplético por departamento del rendimiento predicho para un cultivo seleccionado.
4. **Comparación:** boxplot por departamento + top 10 municipios + media nacional.

### Endpoint adicional

- `POST /predict/batch` — predice para múltiples municipios en una sola llamada (útil para el mapa y para UPRA).
