"""Feature engineering for the BeeBuzz (Mendeley Buzz Pollination) dataset."""
from pathlib import Path

import pandas as pd

TARGET_SPECIES = "morphotype"
TARGET_TYPE = "type"
GROUP_COL = "type"

# Extended features: fundamental frequency + duration + MFCC distribution stats
MFCC_STATS = ["mean.mfc", "sd.mfc", "skew.mfc", "kurt.mfc", "med.mfc"]
FEATURES = ["ffreq", "duration.s"] + [f"{s} {i}" for s in MFCC_STATS for i in range(1, 14)]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parents[1] / path
    return p


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path), sep=";", decimal=",")
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET_SPECIES):
    X = df[FEATURES].copy()
    y = df[target].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups


if __name__ == "__main__":
    df = load("data/raw/buzz_data.csv")
    print(df.shape)
    print(df[TARGET_SPECIES].value_counts())
    X, y, g = split_features_target(df)
    print("n features:", len(FEATURES))
    print("groups:", g.value_counts().to_dict())
    print("nulls:", X.isnull().sum().sum())
