"""Inference helper used by the API."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "model.joblib")
CFG = json.loads((ROOT / "models" / "feature_config.json").read_text())
FEATURES = CFG["features"]
CLASSES = CFG["classes"]

MODEL_TYPE = joblib.load(ROOT / "models" / "model_type.joblib")
CFG_TYPE = json.loads((ROOT / "models" / "feature_config_type.json").read_text())
CLASSES_TYPE = CFG_TYPE["classes"]


def predict_one(features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURES]
    proba = MODEL.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "species": CLASSES[idx],
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, proba)},
    }


def predict_type(features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURES]
    proba = MODEL_TYPE.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "buzz_type": CLASSES_TYPE[idx],
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES_TYPE, proba)},
    }


if __name__ == "__main__":
    sample = {f: 0.0 for f in FEATURES}
    sample["ffreq"] = 239.0
    sample["duration.s"] = 0.304
    print("species:", predict_one(sample))
    print("type:", predict_type(sample))
