# Homework 5 FAQ — Deploying Machine Learning Models (2026 lead-scoring)

Companion to `homework-5.ipynb` and the official task:
`cohorts/2026/homework/05-deployment/homework.md`.
Artifact: frozen `pipeline.bin` (DictVectorizer + LogisticRegression, scikit-learn 1.7.2).
Module notes: `05-deployment` (environment, locked deps, model loading, serving, container).

## Answers (verified)

| Q | Answer |
|---|--------|
| Q1 — uv version (setup check, not graded) | `0.12.12` |
| Q2 — locked scikit-learn version | `1.7.2` |
| Q3 — load model, lead 1 probability | `0.533` |
| Q4 — serve model, lead 2 probability | `0.770` |
| Q5 — Python base-image tag in Dockerfile | `python:3.11.15-slim-bookworm` |
| Q6 — container, lead 2 probability | `0.770` |

## Q1: environment check

Run `uv --version` and record it. It is a setup check, not a graded answer:
your local uv version is not a property of the model. Observed: `uv 0.12.12`.

⚠️ Gotcha: do not confuse the uv version with the Python version (`.python-version` = `3.11.15`)
or with the scikit-learn version (Q2).

## Q2: locked dependency

Open `pyproject.toml` and `uv.lock`. The reference environment pins
`scikit-learn==1.7.2`. Do not regenerate the lockfile for the graded run;
use `uv sync --locked`. Observed install: `scikit-learn==1.7.2`.

⚠️ Gotcha: if you run `uv sync` without `--locked`, uv may resolve a newer
version and your probabilities will differ from the grader.

## Q3: load the model

```python
import pickle
from model import normalize_record

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)

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
prob = float(pipeline.predict_proba([record])[0, 1])  # 0.532994 -> 0.533
```

⚠️ Gotchas:
- Verify the artifact first: `sha256sum pipeline.bin` must equal
  `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`.
- Use `normalize_record` (same as the API) so imputation matches (medians from training set).
- Pickle can execute arbitrary code: only load trusted artifacts; the checksum guarantees integrity.

## Q4: serve the model

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Then POST the second lead:

```bash
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_source":"organic_search","industry":"technology","employment_status":"employed","location":"europe","number_of_courses_viewed":4,"annual_income":80304.0,"interaction_count":7,"lead_score":0.74}'
# -> {"conversion_probability":0.769799,"conversion":true} -> 0.770
```

The response contains a probability in `[0, 1]` and a boolean `conversion` field.
`GET /health` returns `{"status":"ok", "model_sha256":..., "feature_defaults_sha256":...}`.

⚠️ Gotcha: the API loads `pipeline.bin` once at startup; if you replace the
artifact, restart uvicorn.

## Q5: container configuration

The Dockerfile declares `FROM python:3.11.15-slim-bookworm@sha256:...`.
Answer: `python:3.11.15-slim-bookworm`.

⚠️ Gotcha: the displayed size of a local Docker image is not graded (varies by
platform/architecture/Docker version/cached layers). The graded value is the
declared tag in the Dockerfile.

## Q6: run the container

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py   # -> {'conversion_probability': 0.769799, 'conversion': True}
curl -s http://localhost:9696/health
```

Expected: same inference as Q4 (`0.770`), same `0.005` tolerance, health returns
`{"status":"ok", ...}` with the model checksum from `model_metadata.json`.

⚠️ Gotchas:
- Build from the directory containing `uv.lock` and `pipeline.bin`.
- The container uses `uv sync --locked --no-dev`; do not edit the lockfile.
- Free port 9696 before running (stop any local uvicorn first).

## Data-prep / deployment notes

- Missing categoricals → `"NA"`; missing numerics → training-set medians
  (`feature_defaults.json`: courses 2.0, income 55051.0, interactions 4.0, lead_score 0.49).
- `normalize_record` (in `model.py`) applies this policy; reuse it in scripts and API.
- The model is a frozen artifact: do not retrain for the graded run.
- Serialization: `pickle` (standard library). See `notes/Serialización - MOC.md`.

## Verification performed

- `sha256sum pipeline.bin` == expected hash.
- `uv sync --locked` installed scikit-learn 1.7.2.
- `uv run python smoke_test.py` passed (deterministic `0.769799`).
- Local API (Q4) and container (Q6) both returned `0.769799` with matching health checksums.
- Answers rounded to 3 decimals are within the `0.005` grader tolerance.
