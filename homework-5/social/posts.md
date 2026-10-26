# Homework 5 — Posts & Snippets for publishing

Ready for LinkedIn, Instagram and DEV.to. Use `#mlzoomcamp`, tag @AlexeyGrigorev or @DataTalksClub.

---

## Snippet 1 — Load the frozen model (pickle)

```python
import pickle
from model import normalize_record

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)   # DictVectorizer + LogisticRegression

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
record = normalize_record(lead)
prob = float(pipeline.predict_proba([record])[0, 1])
print(round(prob, 3))   # 0.533
```

## Snippet 2 — Serve with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
from model import normalize_record

MODEL_PATH = Path("pipeline.bin")
with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str | None = None
    industry: str | None = None
    employment_status: str | None = None
    location: str | None = None
    number_of_courses_viewed: int | None = None
    annual_income: float | None = None
    interaction_count: int | None = None
    lead_score: float | None = None

@app.post("/predict")
def predict(lead: Lead):
    record = normalize_record(lead.model_dump())
    prob = float(pipeline.predict_proba([record])[0, 1])
    return {"conversion_probability": round(prob, 6),
            "conversion": prob >= 0.5}
```

## Snippet 3 — Minimal reproducible Dockerfile

```dockerfile
FROM python:3.11.15-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY model.py predict.py feature_defaults.json model_metadata.json pipeline.bin ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

## Snippet 4 — Verify the artifact before serving

```bash
sha256sum pipeline.bin
# expected: 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```

## Snippet 5 — Health check

```bash
curl -s http://localhost:9696/health
# {"status":"ok","model_sha256":"1646...","feature_defaults_sha256":"1694..."}
```

---

## Post — LinkedIn

🚀 Completed Module 5 of Machine Learning Zoomcamp 2026!

I packaged a frozen ML model behind a FastAPI service, checked its response,
and ran the same prediction through a reproducible Docker container.

Key takeaways:
- Verify the artifact with sha256 before serving.
- Pin versions with uv.lock; use `uv sync --locked`.
- Serialize the whole pipeline (not just the model) so preprocessing travels with it.
- The container must reproduce the exact same inference as local.

My repo: https://github.com/marcos-Nieves-24/machine-learning-zoomcamp

#mlzoomcamp @Alexey Grigorev

## Post — X / Twitter

Packaged an ML model as an API in Module 5 of #mlzoomcamp.

✅ Checked the pinned model artifact (sha256)
✅ Served predictions with FastAPI
✅ Built and tested the Docker image (same inference as local)

Repo: https://github.com/marcos-Nieves-24/machine-learning-zoomcamp

## Post — DEV.to (short)

Title: "Deploying a frozen ML model with FastAPI + Docker (ML Zoomcamp 2026)"

Body: walk through Q1–Q6, show snippets 1–5, mention the gotchas
(checksum, --locked, normalize_record, port conflicts). Link to repo.
Tag: #machinelearning #fastapi #docker #mlops

---

## Assets

- Cards: `social/out/hw5-*.png`
- Answers video (10s, 4:5): `social/out/hw5-answers-1080x1350.mp4`
- Data-flow video (15s, 16:9): `social/out/hw5-dataflow-1920x1080.mp4`
- Flow diagram v2 (24s, 16:9): `social/out/hw5-flow-v2-1920x1080.mp4` (two scenes: inference + deployment)
- FAQ: `../FAQ.md`
- Notebook: `../homework-5.ipynb`
