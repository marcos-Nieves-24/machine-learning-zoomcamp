---
tags: [ml, serialization, modules, pickle, joblib, cloudpickle]
created: 2026-10-26
related: [[Serialización - Concepto]], [[Serializar un Pipeline]], [[Extensiones de archivo]]
---

# Módulos de serialización

## Comparativa rápida

| Módulo | Cuándo usarlo | Ventajas | Limitaciones |
|---|---|---|---|
| `pickle` | Estándar, cualquier objeto Python | Viene con Python, universal | Menos eficiente con arrays NumPy grandes |
| `joblib` | Pipelines scikit-learn con arrays grandes | Más rápido y eficiente en memoria para NumPy | Requiere instalar `joblib` (viene con sklearn) |
| `cloudpickle` | Funciones/lambdas, closures, código dinámico | Serializa cosas que pickle no puede | Más pesado, pensado para distribución (Spark, Dask) |

## `pickle` (estándar)

```python
import pickle

# Guardar
with open("modelo.pkl", "wb") as f:
    pickle.dump(objeto, f)

# Cargar
with open("modelo.pkl", "rb") as f:
    objeto = pickle.load(f)
```

- Viene incluido en la stdlib (`import pickle`).
- Funciona con cualquier objeto serializable de Python.
- Es lo que usa este homework (`pipeline.bin`).

## `joblib` (recomendado por scikit-learn)

```python
import joblib

# Guardar
joblib.dump(pipeline, "modelo.joblib")

# Cargar
pipeline = joblib.load("modelo.joblib")
```

- Optimizado para arrays NumPy (usa `mmap` y compresión opcional).
- `joblib.dump(obj, "f.joblib", compress=3)` para comprimir.
- Es el formato recomendado por la documentación oficial de scikit-learn para modelos.

## `cloudpickle`

```python
import cloudpickle

# Guardar una función con closure
def make_predictor(threshold):
    def predict(x):
        return x > threshold
    return predict

fn = make_predictor(0.5)
with open("fn.pkl", "wb") as f:
    cloudpickle.dump(fn, f)
```

- Extiende `pickle` para serializar funciones, lambdas y closures.
- Útil en entornos distribuidos (Spark, Ray, Dask) donde el código se envía a workers.
- No es necesario para pipelines scikit-learn normales.

## Regla de oro: consistencia

Lo que importa no es el módulo, sino que **el mismo módulo y la misma versión** se usen para guardar y cargar. Mezclar `pickle` para guardar y `joblib` para cargar (o viceversa) puede fallar.

## En este proyecto

- `train.py` usa `pickle` (implícito vía el artifact `pipeline.bin`).
- `predict.py` carga con `pickle.load`.
- `smoke_test.py` también carga con `pickle.load`.

Todo consistente: mismo módulo, mismas versiones (fijadas en `uv.lock`).

## Ver también

- [[Serialización - Concepto]]
- [[Serializar un Pipeline]]
- [[Extensiones de archivo]]
