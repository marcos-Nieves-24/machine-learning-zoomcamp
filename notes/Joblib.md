---
tags: [ml, serialization, joblib, sklearn]
created: 2026-10-26
related: [[Pickle]], [[Serialización - Concepto]], [[Módulos de serialización]]
---

# Joblib

## Qué es

`joblib` es una librería (incluida con scikit-learn) optimizada para serializar objetos que contienen **arrays NumPy grandes**. Es la recomendación oficial de scikit-learn para guardar modelos.

## API básica

```python
import joblib

# Guardar
joblib.dump(pipeline, "modelo.joblib")

# Cargar
pipeline = joblib.load("modelo.joblib")
```

## Ventajas sobre pickle

- Usa `mmap` para cargar arrays sin copiarlos a RAM.
- Soporta compresión: `joblib.dump(obj, "f.joblib", compress=3)`.
- Más rápido y eficiente en memoria para modelos con muchos coeficientes/arrays.

## Cuándo usarlo

- Pipelines scikit-learn con arrays NumPy grandes.
- Cuando el modelo ocupa mucho espacio en disco/RAM.
- Cuando querés compresión.

## En este proyecto

Este homework usa `pickle` (no `joblib`) porque el modelo es chico (~1.9 KB). Para modelos más grandes, `joblib` sería la mejor opción.

## Ver también

- [[Pickle]] — estándar de Python.
- [[Serialización - Concepto]]
- [[Módulos de serialización]]
- [[Extensiones de archivo]]
