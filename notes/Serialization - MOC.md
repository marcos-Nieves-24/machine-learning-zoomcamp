---
tags: [ml, serialization, obsidian, knowledge-base]
created: 2026-10-26
related: [[Pickle]], [[Joblib]], [[Serializing a Pipeline]], [[File Extensions]]
---

# Serialization — Map of Contents (MOC)

> Knowledge notes on ML model serialization, Obsidian style.
> Origin: Homework 5 (ML Zoomcamp 2026) — frozen artifact `pipeline.bin`.

## What is serialization?

**Serialization** is the process of converting an in-memory object (a trained model, a pipeline, a data structure) into a byte sequence that can be saved to disk and reconstructed later.

- See: [[Serialization - Concept]]
- See: [[Pickle]]
- See: [[Joblib]]

## Topics

1. [[Serialization - Concept]] — what it is, how it works, why it is used.
2. [[Serialization Modules]] — `pickle`, `joblib`, `cloudpickle`, when to use each.
3. [[Serializing a Pipeline]] — full flow: train → save → load → predict.
4. [[File Extensions]] — `.pkl`, `.bin`, `.joblib`, `.sav`: what they really mean.

## Project connections

- Frozen artifact: `homework-5/pipeline.bin` (SHA256 `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`).
- Training script: `homework-5/train.py`.
- Serving: `homework-5/predict.py` + `homework-5/model.py`.
- Container: `homework-5/Dockerfile` (pins Python + uv + lockfile + artifact).

## Golden rules

- Serialize the **whole pipeline**, not just the model, so preprocessing travels with it.
- Save **imputation defaults** separately (`feature_defaults.json`).
- Pin **library versions** (`uv.lock`, `pyproject.toml`).
- Verify the **checksum** of the artifact before serving.
- Only load pickles from trusted sources (pickle can execute arbitrary code).
- The **file extension does not matter**: what matters are the bytes and the library that produced them.
