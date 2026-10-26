---
tags: [ml, serialization, modules, pickle, joblib, cloudpickle]
created: 2026-10-26
related: [[Serialization - Concept]], [[Serializing a Pipeline]], [[File Extensions]]
---

# Serialization Modules

## Quick comparison

| Module | When to use | Advantages | Limitations |
|---|---|---|---|
| `pickle` | Standard, any Python object | Ships with Python, universal | Less efficient with large NumPy arrays |
| `joblib` | scikit-learn pipelines with large arrays | Faster and more memory-efficient for NumPy | Requires installing `joblib` (ships with sklearn) |
| `cloudpickle` | Functions/lambdas, closures, dynamic code | Serializes things pickle cannot | Heavier, aimed at distribution (Spark, Dask) |

## `pickle` (standard)

```python
import pickle

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(obj, f)

# Load
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)
```

- Ships in the standard library (`import pickle`).
- Works with any serializable Python object.
- This is what the homework uses (`pipeline.bin`).

## `joblib` (recommended by scikit-learn)

```python
import joblib

# Save
joblib.dump(pipeline, "model.joblib")

# Load
pipeline = joblib.load("model.joblib")
```

- Optimized for NumPy arrays (uses `mmap` and optional compression).
- `joblib.dump(obj, "f.joblib", compress=3)` to compress.
- This is the format recommended by the official scikit-learn documentation for models.

## `cloudpickle`

```python
import cloudpickle

# Save a function with a closure
def make_predictor(threshold):
    def predict(x):
        return x > threshold
    return predict

fn = make_predictor(0.5)
with open("fn.pkl", "wb") as f:
    cloudpickle.dump(fn, f)
```

- Extends `pickle` to serialize functions, lambdas, and closures.
- Useful in distributed environments (Spark, Ray, Dask) where code is sent to workers.
- Not needed for normal scikit-learn pipelines.

## Golden rule: consistency

What matters is not the module, but that **the same module and the same version** are used to save and to load. Mixing `pickle` to save and `joblib` to load (or vice versa) can fail.

## In this project

- `train.py` uses `pickle` (implicitly, via the `pipeline.bin` artifact).
- `predict.py` loads with `pickle.load`.
- `smoke_test.py` also loads with `pickle.load`.

Consistent throughout: same module, same versions (pinned in `uv.lock`).

## See also

- [[Serialization - Concept]]
- [[Serializing a Pipeline]]
- [[File Extensions]]
