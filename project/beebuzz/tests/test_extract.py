"""Tests for scripts/extract_features.py."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_features.py"
SR = 44100


def _tone(path: Path, freq=250.0, dur=0.5, sr=SR):
    import soundfile as sf
    t = np.arange(int(sr * dur)) / sr
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), y, sr)


def test_synthetic_tone_ffreq(tmp_path):
    wav = tmp_path / "tone.wav"
    _tone(wav, freq=250.0, dur=0.5)
    out = tmp_path / "feat.csv"
    r = subprocess.run([sys.executable, str(SCRIPT), str(wav), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out, sep=";", decimal=",")
    assert abs(df.loc[0, "ffreq"] - 250.0) < 15.0
    assert abs(df.loc[0, "duration.s"] - 0.5) < 0.05


def test_output_format_matches_dashboard(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from features import FEATURES as F
    wav = tmp_path / "tone.wav"
    _tone(wav)
    out = tmp_path / "feat.csv"
    subprocess.run([sys.executable, str(SCRIPT), str(wav), "--out", str(out)], check=True,
                   capture_output=True)
    df = pd.read_csv(out, sep=";", decimal=",")
    for f in F:
        assert f in df.columns, f"missing {f}"
    assert df["source_file"].iloc[0] == "tone.wav"


def test_rejects_silence(tmp_path):
    import soundfile as sf
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(SR // 2, dtype=np.float32), SR)
    out = tmp_path / "feat.csv"
    r = subprocess.run([sys.executable, str(SCRIPT), str(wav), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "silence" in r.stderr.lower()


def test_rejects_non_wav(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    out = tmp_path / "feat.csv"
    r = subprocess.run([sys.executable, str(SCRIPT), str(txt), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_rejects_short_clip(tmp_path):
    wav = tmp_path / "short.wav"
    _tone(wav, dur=0.02)
    out = tmp_path / "feat.csv"
    r = subprocess.run([sys.executable, str(SCRIPT), str(wav), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "short" in r.stderr.lower()
