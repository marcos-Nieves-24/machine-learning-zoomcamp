---
tags: [ml, serialization, pickle, python]
created: 2026-10-26
related: [[Serialización - Concepto]], [[Módulos de serialización]], [[Joblib]]
---

# Pickle

## Qué es

`pickle` es un módulo de la **stdlib de Python** para serializar y deserializar objetos Python. Viene incluido, no hay que instalarlo.

## API básica

```python
import pickle

# Serializar (objeto → bytes en disco)
with open("archivo.pkl", "wb") as f:
    pickle.dump(objeto, f)

# Deserializar (bytes en disco → objeto)
with open("archivo.pkl", "rb") as f:
    objeto = pickle.load(f)
```

Modos de apertura: `"wb"` (write binary) y `"rb"` (read binary).

## Cuándo usarlo

- Cualquier objeto Python serializable.
- Pipelines scikit-learn chicos/medianos.
- Cuando querés cero dependencias extra.

## Advertencias

1. **Seguridad**: `pickle.load` puede ejecutar código arbitrario. Solo cargá pickles de fuentes confiables. En este proyecto, el checksum SHA256 del `pipeline.bin` garantiza integridad.
2. **Versiones**: un pickle creado con scikit-learn 1.7.2 puede fallar con 1.6.1. Pinneá versiones (`uv.lock`).
3. **Eficiencia**: para arrays NumPy muy grandes, `joblib` es más eficiente.

## En este proyecto

- `pipeline.bin` fue creado con `pickle.dump` (en `train.py`).
- `predict.py` y `smoke_test.py` lo cargan con `pickle.load`.
- Consistente en todo el flujo.

## Ver también

- [[Joblib]] — alternativa para arrays grandes.
- [[Serialización - Concepto]]
- [[Módulos de serialización]]
- [[Extensiones de archivo]]
