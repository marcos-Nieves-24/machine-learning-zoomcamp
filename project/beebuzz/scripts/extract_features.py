"""Extract the 67 BeeBuzz acoustic features from .wav files.

Produces a CSV in the exact format the dashboard accepts
(sep=';', decimal=','), so the output can be uploaded directly
at the Predict tab.

Usage:
    python scripts/extract_features.py audio1.wav audio2.wav --out features.csv
    python scripts/extract_features.py ./recordings/ --out features.csv

Pipeline (matches the paper's setup):
    - resample to 44100 Hz, mono
    - 13 MFCC (n_fft=2048, hop_length=512)
    - fundamental frequency via librosa.yin (median over voiced frames)
    - per-MFCC summary stats: mean, sd, skew, kurtosis, median

Caveat (important): the training CSV was produced with the paper's R
pipeline, whose MFCC magnitudes differ systematically from librosa's
(verified: librosa mean.mfc values fall outside the training ranges,
while ffreq/duration match). So this script outputs the correct schema
for pipeline testing, but its MFCCs are NOT on the training scale —
predictions on librosa features need recalibration (capstone: re-extract
the training .wavs with this script and retrain, or per-feature
standardization). Physical features (ffreq, duration.s) transfer directly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SR = 44100
N_MFCC = 13
N_FFT = 2048
HOP = 512
MIN_DURATION_S = 0.05
MIN_SR = 8000

FEATURES = (
    ["ffreq", "duration.s"]
    + [f"mean.mfc {i}" for i in range(1, 14)]
    + [f"sd.mfc {i}" for i in range(1, 14)]
    + [f"skew.mfc {i}" for i in range(1, 14)]
    + [f"kurt.mfc {i}" for i in range(1, 14)]
    + [f"med.mfc {i}" for i in range(1, 14)]
)


def _skew(x):
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float((((x - m) / s) ** 3).mean())


def _kurt(x):
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float((((x - m) / s) ** 4).mean())


def extract_one(path: Path):
    """Returns (row_dict, warnings). Raises ValueError on invalid input."""
    import librosa

    if path.suffix.lower() != ".wav":
        raise ValueError(f"not a .wav file: {path}")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr < MIN_SR:
        raise ValueError(f"sample rate {sr} Hz below minimum {MIN_SR} Hz: {path}")
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    duration = len(y) / SR
    if duration < MIN_DURATION_S:
        raise ValueError(f"clip too short ({duration:.3f}s < {MIN_DURATION_S}s): {path}")
    if np.all(y == 0) or np.sqrt(np.mean(y ** 2)) < 1e-8:
        raise ValueError(f"silence (RMS ~ 0): {path}")

    warnings = []
    if duration < 0.15:
        warnings.append("short clip: stats estimated from few frames")

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    # fundamental frequency: yin over voiced frames, median
    try:
        f0 = librosa.yin(y, fmin=50, fmax=1000, sr=SR, hop_length=HOP)
        voiced = f0[(f0 > 0) & np.isfinite(f0)]
        ffreq = float(np.median(voiced)) if len(voiced) else float(np.median(f0[np.isfinite(f0)]))
    except Exception:
        spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        ffreq = float(freqs[np.argmax(spec.mean(axis=1))])
        warnings.append("f0 fallback to spectral peak")

    row = {"ffreq": round(ffreq, 4), "duration.s": round(float(duration), 4)}
    for i in range(N_MFCC):
        col = mfcc[i]
        row[f"mean.mfc {i+1}"] = round(float(col.mean()), 6)
        row[f"sd.mfc {i+1}"] = round(float(col.std()), 6)
        row[f"skew.mfc {i+1}"] = round(_skew(col), 6)
        row[f"kurt.mfc {i+1}"] = round(_kurt(col), 6)
        row[f"med.mfc {i+1}"] = round(float(np.median(col)), 6)
    row["source_file"] = path.name
    return row, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract BeeBuzz features from .wav files.")
    ap.add_argument("inputs", nargs="+", help=".wav files or directories")
    ap.add_argument("--out", default="features.csv", help="output CSV path")
    args = ap.parse_args(argv)

    files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip (not found): {inp}", file=sys.stderr)
    if not files:
        print("no .wav files found", file=sys.stderr)
        return 1

    rows = []
    failed = 0
    for f in files:
        try:
            row, warnings = extract_one(f)
            rows.append(row)
            msg = f"ok: {f.name} (ffreq={row['ffreq']} Hz, dur={row['duration.s']}s)"
            if warnings:
                msg += " [" + "; ".join(warnings) + "]"
            print(msg)
        except ValueError as e:
            failed += 1
            print(f"reject: {e}", file=sys.stderr)

    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in FEATURES if c in df.columns] + ["source_file"]
        df = df[cols]
        df.to_csv(args.out, index=False, sep=";", decimal=",")
        print(f"wrote {len(rows)} rows -> {args.out} (rejected: {failed})")
    else:
        print("nothing extracted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
