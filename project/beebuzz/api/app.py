"""FastAPI service for BeeBuzz (bee species + buzz type classification)."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

from predict import FEATURES, predict_one, predict_type

# Biologically plausible ranges (from training data, with margin).
# ffreq: bee buzz fundamental ~50-1000 Hz. duration: positive, < 5 s.
# MFCC stats: unbounded in theory; reject absurd values to catch unit errors.
FIELD_CONSTRAINTS = {"ffreq": (50.0, 1000.0), "duration.s": (0.0, 5.0)}
MFCC_ABS_MAX = 500.0

_fields = {}
for f in FEATURES:
    if f in FIELD_CONSTRAINTS:
        lo, hi = FIELD_CONSTRAINTS[f]
        _fields[f] = (float, ... )
    else:
        _fields[f] = (float, ...)
PredictRequest = create_model("PredictRequest", **_fields)

MODELS_OK = {"species": False, "type": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Explicit startup: verify models load before serving traffic.
    try:
        import predict as _p
        _ = _p.MODEL
        _ = _p.MODEL_TYPE
        MODELS_OK["species"] = True
        MODELS_OK["type"] = True
    except Exception as e:
        MODELS_OK["species"] = False
        MODELS_OK["type"] = False
        print(f"MODEL LOAD FAILED: {e}")
    yield


app = FastAPI(title="BeeBuzz Classifier", version="0.2.0", lifespan=lifespan)


def _validate_ranges(d: dict):
    for f, (lo, hi) in FIELD_CONSTRAINTS.items():
        v = d.get(f)
        if v is not None and not (lo <= v <= hi):
            raise HTTPException(
                status_code=422,
                detail=f"{f}={v} out of plausible range [{lo}, {hi}]. Check units (Hz, seconds).")
    for f in FEATURES:
        if f.startswith(("mean.", "sd.", "skew.", "kurt.", "med.")):
            v = d.get(f)
            if v is not None and abs(v) > MFCC_ABS_MAX:
                raise HTTPException(
                    status_code=422,
                    detail=f"{f}={v} implausible (|v| > {MFCC_ABS_MAX}). Check feature extraction.")


@app.get("/health")
def health():
    return {"status": "ok" if all(MODELS_OK.values()) else "degraded",
            "models": MODELS_OK, "features": FEATURES,
            "calibration_note": "confidence is RF vote frequency, not calibrated probability (Brier 0.084, ECE 0.31 on holdout)"}


@app.post("/predict")
def predict(req: PredictRequest):
    d = req.model_dump()
    _validate_ranges(d)
    if not MODELS_OK["species"]:
        # lazy check in case lifespan did not run (e.g. tests)
        try:
            import predict as _p
            _ = _p.MODEL
        except Exception:
            raise HTTPException(status_code=503, detail="species model unavailable")
    return predict_one(d)


@app.post("/predict/type")
def predict_buzz_type(req: PredictRequest):
    d = req.model_dump()
    _validate_ranges(d)
    return predict_type(d)
