---
tags: [ml, serialization, concept]
created: 2026-10-26
related: [[Pickle]], [[Joblib]], [[Serializing a Pipeline]]
---

# Serialization — Concept

## Definition

**Serialization** is the process of converting an in-memory object (a trained model, a pipeline, a data structure) into a byte sequence that can be:

- Saved to disk.
- Transmitted over a network.
- Reconstructed later (deserialization) into an identical object.

## Analogy

Think of it as **freezing and thawing** a model:

- **Train** = cook the dish.
- **Serialize** = freeze it (save the exact state).
- **Deserialize** = thaw it (recover the same dish, ready to serve).

## How it works (high level)

A trained model is a **graph of Python objects**:

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

Serializing = walk that graph and write every object and its attributes as bytes.
Deserializing = read the bytes and rebuild the identical graph, with the same learned values.

## Why it is used

1. **No retraining**: training can take hours; serializing takes milliseconds.
2. **Separate training from serving**: train in one environment (GPU, big data) and serve in another (lightweight container).
3. **Reproducibility**: the frozen artifact is the contract; the grader compares against it.
4. **Versioning**: each model is a file with a checksum; you can version them as artifacts.

## Minimal example

```python
import pickle

# Object in memory
pipeline = train_pipeline(X, y)   # has learned coefficients

# Serialize
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)         # bytes -> disk

# Deserialize (another script, another day)
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)        # disk -> identical object

proba = pipeline.predict_proba([lead])  # works without retraining
```

## Security warning

`pickle.load` **can execute arbitrary code**. Only deserialize files from trusted sources. In this project, the SHA256 checksum of `pipeline.bin` guarantees integrity.

## See also

- [[Pickle]] — standard Python module.
- [[Joblib]] — alternative recommended by scikit-learn.
- [[Serializing a Pipeline]] — full flow.
- [[File Extensions]] — `.pkl` vs `.bin`.
