"""FastAPI service for BeeBuzz (bee species + buzz type classification)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from pydantic import BaseModel, create_model

from predict import FEATURES, predict_one, predict_type

PredictRequest = create_model("PredictRequest", **{f: (float, ...) for f in FEATURES})

app = FastAPI(title="BeeBuzz Classifier", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}


@app.post("/predict")
def predict(req: PredictRequest):
    return predict_one(req.model_dump())


@app.post("/predict/type")
def predict_buzz_type(req: PredictRequest):
    return predict_type(req.model_dump())
