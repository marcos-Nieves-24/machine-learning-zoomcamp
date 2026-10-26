---
tags: [ml, serialization, concept]
created: 2026-10-26
related: [[Pickle]], [[Joblib]], [[Serializar un Pipeline]]
---

# Serialización — Concepto

## Definición

La **serialización** es el proceso de convertir un objeto que vive en memoria (un modelo entrenado, un pipeline, una estructura de datos) en una secuencia de bytes que se puede:

- Guardar en disco.
- Transmitir por red.
- Reconstruir después (deserialización) en un objeto idéntico.

## Analogía

Pensalo como **congelar y descongelar** un modelo:

- **Entrenar** = cocinar el plato.
- **Serializar** = congelarlo (guardar el estado exacto).
- **Deserializar** = descongelarlo (recuperar el mismo plato, listo para servir).

## Cómo funciona (a alto nivel)

Un modelo entrenado es un **grafo de objetos Python**:

```
Pipeline
├── DictVectorizer
│   ├── vocabulary_ (dict)
│   └── feature_names_ (list)
└── LogisticRegression
    ├── coef_ (ndarray)
    ├── intercept_ (ndarray)
    ├── classes_ (ndarray)
    └── ...
```

Serializar = recorrer ese grafo y escribir cada objeto y sus atributos en bytes.  
Deserializar = leer los bytes y reconstruir el grafo idéntico, con los mismos valores aprendidos.

## Por qué se usa

1. **No reentrenar**: entrenar puede tardar horas; serializar toma milisegundos.
2. **Separar entrenamiento de serving**: entrenás en un entorno (con GPU, datos grandes) y servís en otro (contenedor liviano).
3. **Reproducibilidad**: el artifact congelado es el contrato; el grader compara contra él.
4. **Versionado**: cada modelo es un archivo con checksum; podés versionarlos como artifacts.

## Ejemplo mínimo

```python
import pickle

# Objeto en memoria
pipeline = entrenar_pipeline(X, y)   # tiene coeficientes aprendidos

# Serializar
with open("modelo.pkl", "wb") as f:
    pickle.dump(pipeline, f)         # bytes → disco

# Deserializar (otro script, otro día)
with open("modelo.pkl", "rb") as f:
    pipeline = pickle.load(f)        # disco → objeto idéntico

proba = pipeline.predict_proba([lead])  # funciona sin reentrenar
```

## Advertencia de seguridad

`pickle.load` **puede ejecutar código arbitrario**. Solo deserializá archivos de fuentes confiables. En este proyecto, el checksum SHA256 del `pipeline.bin` garantiza que el artifact no fue manipulado.

## Ver también

- [[Pickle]] — módulo estándar de Python.
- [[Joblib]] — alternativa recomendada por scikit-learn.
- [[Serializar un Pipeline]] — flujo completo.
- [[Extensiones de archivo]] — `.pkl` vs `.bin`.
