---
tags: [ml, serialization, pipeline, sklearn, workflow]
created: 2026-10-26
related: [[Serialization - Concept]], [[Serialization Modules]], [[File Extensions]]
---

# Serializing a Pipeline

## Full flow: train → save → load → predict

### 1. Train

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

dv = DictVectorizer(sparse=False)
model = LogisticRegression(solver="liblinear", C=1.0)
pipeline = make_pipeline(dv, model)

pipeline.fit(X_train, y_train)
```

### 2. Save (serialize)

```python
import pickle

with open("pipeline.bin", "wb") as f:
    pickle.dump(pipeline, f)
```

### 3. Load (deserialize)

```python
import pickle

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)
```

### 4. Predict

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

## Why serialize the whole pipeline

If you serialize only the model (without the vectorizer), at serving time you have to:

1. Replicate the one-hot encoding by hand.
2. Make sure the column order matches.
3. Impute missing values with the training-set medians.

That is fragile and error-prone. Serializing the full `Pipeline`, the object already knows how to transform a raw lead into the matrix the model expects.

## What to save besides the model

| File | Content | Why |
|---|---|---|
| `pipeline.bin` | Trained model + vectorizer | The core of serving |
| `feature_defaults.json` | Imputation medians | To apply the same missing-value policy |
| `model_metadata.json` | Checksums, versions, dataset | Traceability and verification |
| `uv.lock` / `pyproject.toml` | Pinned versions | Environment reproducibility |

## Serving pattern (as in this project)

```python
# predict.py
import pickle
from pathlib import Path

MODEL_PATH = Path("pipeline.bin")
with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)   # loaded ONCE at startup

@app.post("/predict")
def predict(lead: Lead):
    record = normalize_record(lead.model_dump())  # impute medians
    prob = float(pipeline.predict_proba([record])[0, 1])
    return {"conversion_probability": round(prob, 6), "conversion": prob >= 0.5}
```

## Checklist before serving

- [ ] Artifact checksum matches the expected value.
- [ ] scikit-learn/numpy versions are identical between training and serving.
- [ ] `normalize_record` applies the same training-set medians.
- [ ] Smoke test passes (`predict_proba` is deterministic).
- [ ] Container reproduces the same result (Q6 of the homework).

## See also

- [[Serialization - Concept]]
- [[Serialization Modules]]
- [[File Extensions]]
- [[Pickle]]
- [[Joblib]]
