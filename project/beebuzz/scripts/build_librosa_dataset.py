"""Build the librosa version of the BeeBuzz dataset.

For every row in data/raw/buzz_data.csv, downloads the matching .wav from the
companion GitHub repo (cached in data/raw/all_audios/), extracts the 67 features
with scripts/extract_features.py, and attaches the original labels.
Output: data/processed/buzz_librosa.csv (same schema + morphotype/type).

Usage:
    python scripts/build_librosa_dataset.py
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_features import extract_one
from features import FEATURES

RAW_CSV = ROOT / "data" / "raw" / "buzz_data.csv"
AUDIO_DIR = ROOT / "data" / "raw" / "all_audios"
OUT_CSV = ROOT / "data" / "processed" / "buzz_librosa.csv"
BASE_URL = ("https://raw.githubusercontent.com/Caarvelos/"
            "Bee-acoustics-classification/main/data/all_audios")


def main():
    from features import load
    df = load(str(RAW_CSV))
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        fn = r["audio_files"]
        dest = AUDIO_DIR / fn
        if not dest.exists():
            print(f"downloading {fn} ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{fn}", dest)
        try:
            feats, warnings = extract_one(dest)
        except ValueError as e:
            print(f"SKIP {fn}: {e}")
            continue
        row = {f: feats[f] for f in FEATURES}
        row["morphotype"] = r["morphotype"]
        row["type"] = r["type"]
        row["audio_files"] = fn
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, sep=";", decimal=",")
    print(f"wrote {len(out)} rows -> {OUT_CSV}")
    print(out["morphotype"].value_counts().to_dict())


if __name__ == "__main__":
    raise SystemExit(main())
