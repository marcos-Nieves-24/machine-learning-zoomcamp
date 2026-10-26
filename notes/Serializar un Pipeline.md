---
tags: [ml, serialization, pipeline, sklearn, workflow]
created: 2026-10-26
related: [[Serialización - Concepto]], [[Módulos de serialización]], [[Extensiones de archivo]]
---

# Serializar un Pipeline

## Flujo completo: entrenar → guardar → cargar → predecir

### 1. Entrenar

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

dv = DictVectorizer(sparse=False)
model = LogisticRegression(solver="liblinear", C=1.0)
pipeline = make_pipeline(dv, model)

pipeline.fit(X_train, y_train)
```

### 2. Guardar (serializar)

```python
import pickle

with open("pipeline.bin", "wb") as f:
    pickle.dump(pipeline, f)
```

### 3. Cargar (deserializar)

```python
import pickle

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)
```

### 4. Predecir

```python
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

prob = pipeline.predict_proba([lead])[0, 1]
print(prob)  # 0.532994
```

## Por qué serializar el pipeline completo

Si serializás solo el modelo (sin el vectorizador), al servir tenés que:

1. Replicar a mano el one-hot encoding.
2. Asegurarte de que el orden de las columnas coincida.
3. Imputar missing values con las medianas del training set.

Eso es frágil y propenso a bugs. Serializando el `Pipeline` completo, el objeto ya sabe cómo transformar un lead crudo en la matriz que el modelo espera.

## Lo que hay que guardar además del modelo

| Archivo | Contenido | Por qué |
|---|---|---|
| `pipeline.bin` | Modelo + vectorizador entrenados | El corazón del serving |
| `feature_defaults.json` | Medianas de imputación | Para aplicar la misma política de missing values |
| `model_metadata.json` | Checksums, versiones, dataset | Trazabilidad y verificación |
| `uv.lock` / `pyproject.toml` | Versiones pinneadas | Reproducibilidad del entorno |

## Patrón de serving (como en este proyecto)

```python
# predict.py
import pickle
from pathlib import Path

MODEL_PATH = Path("pipeline.bin")
with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)   # se carga UNA vez al arrancar

@app.post("/predict")
def predict(lead: Lead):
    record = normalize_record(lead.model_dump())  # imputa medianas
    prob = float(pipeline.predict_proba([record])[0, 1])
    return {"conversion_probability": round(prob, 6), "conversion": prob >= 0.5}
```

## Checklist antes de servir

- [ ] Checksum del artifact coincide con el esperado.
- [ ] Versiones de scikit-learn/numpy iguales entre entrenamiento y serving.
- [ ] `normalize_record` aplica las mismas medianas del training set.
- [ ] Smoke test pasa (`predict_proba` determinístico).
- [ ] Container reproduce el mismo resultado (Q6 del homework).

## Ver también

- [[Serialización - Concepto]]
- [[Módulos de serialización]]
- [[Extensiones de archivo]]
- [[Pickle]]
- [[Joblib]]
