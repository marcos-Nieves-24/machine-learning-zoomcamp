"""Smoke tests for the API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {f: 0.0 for f in __import__("features", fromlist=["FEATURES"]).FEATURES}
    payload["ffreq"] = 239.0
    payload["duration.s"] = 0.304
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "species" in body
    assert "confidence" in body


def test_predict_type():
    payload = {f: 0.0 for f in __import__("features", fromlist=["FEATURES"]).FEATURES}
    payload["ffreq"] = 239.0
    payload["duration.s"] = 0.304
    r = client.post("/predict/type", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "buzz_type" in body
    assert "confidence" in body
