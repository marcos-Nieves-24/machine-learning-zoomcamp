---
tags: [ml, serialization, obsidian, knowledge-base]
created: 2026-10-26
related: [[Pickle]], [[Joblib]], [[Serializar un Pipeline]], [[Extensiones de archivo]]
---

# Serialización — Mapa de Contenidos (MOC)

> Notas de conocimiento sobre serialización de modelos ML, estilo Obsidian.  
> Origen: Homework 5 (ML Zoomcamp 2026) — artifact congelado `pipeline.bin`.

## ¿Qué es la serialización?

La **serialización** es el proceso de convertir un objeto en memoria (un modelo entrenado, un pipeline, una estructura de datos) en una secuencia de bytes que se puede guardar en disco y reconstruir después.

- Ver: [[Serialización - Concepto]]
- Ver: [[Pickle]]
- Ver: [[Joblib]]

## Temas

1. [[Serialización - Concepto]] — qué es, cómo funciona, por qué se usa.
2. [[Módulos de serialización]] — `pickle`, `joblib`, `cloudpickle`, cuándo usar cada uno.
3. [[Serializar un Pipeline]] — flujo completo: entrenar → guardar → cargar → predecir.
4. [[Extensiones de archivo]] — `.pkl`, `.bin`, `.joblib`, `.sav`: qué significan realmente.

## Conexiones con el proyecto

- Artifact congelado: `homework-5/pipeline.bin` (SHA256 `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`).
- Training script: `homework-5/train.py`.
- Serving: `homework-5/predict.py` + `homework-5/model.py`.
- Container: `homework-5/Dockerfile` (congela Python + uv + lockfile + artifact).

## Reglas de oro

- Serializá el **pipeline completo**, no solo el modelo, para no replicar el preprocesamiento a mano.
- Guardá los **defaults de imputación** aparte (`feature_defaults.json`).
- Pinneá las **versiones** de librerías (`uv.lock`, `pyproject.toml`).
- Verificá el **checksum** del artifact antes de servir.
- Solo cargá pickles de fuentes confiables (pickle puede ejecutar código arbitrario).
- La extensión del archivo **no importa**: importan los bytes y la librería que los produjo.
