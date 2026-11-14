"""Full EDA for BeeBuzz: distributions, floral vs flight, correlations, PCA, metadata, leakage checks."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features import FEATURES, TARGET_SPECIES, TARGET_TYPE, load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "buzz_data.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = load(str(DATA))
    print(f"shape: {df.shape}")
    print(f"nulls: {df.isnull().sum().sum()}")

    # 1. Class distribution
    print("\n=== Species distribution ===")
    print(df[TARGET_SPECIES].value_counts())
    print("\n=== Type distribution ===")
    print(df[TARGET_TYPE].value_counts())
    print("\n=== Species x Type ===")
    print(pd.crosstab(df[TARGET_SPECIES], df[TARGET_TYPE]))

    # 2. ffreq by species
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=TARGET_SPECIES, y="ffreq", hue=TARGET_TYPE, ax=ax)
    ax.set_title("Fundamental frequency by species and buzz type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_ffreq.png", dpi=150)
    plt.close()
    print("saved eda_ffreq.png")

    # 3. duration by species
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=TARGET_SPECIES, y="duration.s", hue=TARGET_TYPE, ax=ax)
    ax.set_title("Duration by species and buzz type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_duration.png", dpi=150)
    plt.close()
    print("saved eda_duration.png")

    # 4. MFCC mean heatmap (mean of mean.mfc per species)
    mfcc_means = [f"mean.mfc {i}" for i in range(1, 14)]
    heat = df.groupby(TARGET_SPECIES)[mfcc_means].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mean MFCC profile per species")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_mfcc_heatmap.png", dpi=150)
    plt.close()
    print("saved eda_mfcc_heatmap.png")

    # 5. Correlation matrix of the 67 features
    X = df[FEATURES]
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title("Feature correlation matrix (67 features)")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_corr.png", dpi=150)
    plt.close()
    # top redundant pairs
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask).stack()
    high = high[high.abs() > 0.9].sort_values(ascending=False)
    print(f"\npairs with |corr| > 0.9: {len(high)}")
    print(high.head(10).to_string())

    # 6. PCA (2D) colored by species, shaped by type
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
    pdf = pd.DataFrame(Z, columns=["PC1", "PC2"])
    pdf[TARGET_SPECIES] = df[TARGET_SPECIES].values
    pdf[TARGET_TYPE] = df[TARGET_TYPE].values
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=pdf, x="PC1", y="PC2", hue=TARGET_SPECIES, style=TARGET_TYPE, s=70, ax=ax)
    ax.set_title(f"PCA of acoustic features (var: {pca.explained_variance_ratio_.round(2)})")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda_pca.png", dpi=150)
    plt.close()
    print("saved eda_pca.png")

    # 7. Metadata / leakage checks
    print("\n=== Metadata checks ===")
    print(f"audio_files unique: {df['audio_files'].nunique()} / rows: {len(df)}")
    print(f"sample_numer unique: {df['sample_numer'].nunique()}")
    print("sample_numer group sizes:")
    print(df.groupby('sample_numer').size().value_counts().to_string())
    # prefix check: B_ vs V_
    df['_prefix'] = df['audio_files'].str[0]
    print("\nfile prefix x type:")
    print(pd.crosstab(df['_prefix'], df[TARGET_TYPE]))
    print("\nfile prefix x species:")
    print(pd.crosstab(df['_prefix'], df[TARGET_SPECIES]))
    # date / local
    print("\ndate values:", df['date'].unique()[:12])
    print("local values:", df['local'].unique())

    # 8. Per-feature discriminative power (ANOVA F approx via group means)
    print("\n=== Feature means per species (top discriminative) ===")
    means = df.groupby(TARGET_SPECIES)[FEATURES].mean()
    stds = df.groupby(TARGET_SPECIES)[FEATURES].std()
    # coefficient of variation across species means, normalized by within-species std
    spread = means.std() / (stds.mean() + 1e-9)
    print(spread.sort_values(ascending=False).head(15).to_string())

    print("\nEDA done. Figures in reports/")


if __name__ == "__main__":
    main()
