---
tags: [ml, serialization, file-extensions, conventions]
created: 2026-10-26
related: [[Serialization - Concept]], [[Serialization Modules]], [[Serializing a Pipeline]]
---

# File Extensions

## The fundamental truth

**The extension does not change the content.** Python does not look at the extension; it looks at the file bytes.

You can rename `pipeline.bin` to `pipeline.pkl` and `pickle.load` will load it just the same:

```bash
cp pipeline.bin pipeline.pkl
python -c "import pickle; print(pickle.load(open('pipeline.pkl','rb')))"
```

The extension is just a **label for humans**.

## Common extensions

| Extension | Conventional meaning | Typical library |
|---|---|---|
| `.pkl` | "This is a pickle" | `pickle` |
| `.bin` | "Generic binary file" | any |
| `.joblib` | "Serialized with joblib" | `joblib` |
| `.sav` | Used by scikit-learn/SPSS in some contexts | varies |

## Why the homework uses `.bin`

It is a convention of the course (and of many deployments) to indicate a **"frozen binary artifact"**, without tying the name to a specific library. The `Dockerfile` and `predict.py` only reference the file name, not the extension.

## What really determines whether a file can be loaded

It is not the extension, but:

1. **Which library serialized it** (`pickle` vs `joblib` vs `cloudpickle`).
2. **Which version** of that library and of scikit-learn/NumPy was used at serialization time.
3. **What object** it contains (a `Pipeline`, a standalone `LogisticRegression`, etc.).

## Practical rule

Use whatever extension you prefer, but be **consistent** in your project. If you change the extension, update the references in `predict.py`, `Dockerfile`, and scripts. The SHA256 checksum does not change because the content does not change.

## See also

- [[Serialization - Concept]]
- [[Serialization Modules]]
- [[Serializing a Pipeline]]
- [[Pickle]]
- [[Joblib]]
