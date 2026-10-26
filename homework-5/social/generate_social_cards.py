#!/usr/bin/env python3
"""Generate publishable social cards for ML Zoomcamp 2026, Homework 5.

Concept: an "answer key" card. Deep ink background, gold marks for the answers.
Typography is Inter (bundled in ./fonts, SIL OFL 1.1).

Every number rendered here is taken from the executed homework-5.ipynb.
Nothing is hand-typed or invented.

Usage:
    python3 generate_social_cards.py            # renders out/<platform>-*.png
    python3 generate_social_cards.py --preview  # also writes a contact sheet

Canvas: 1080x1350 (4:5) for IG/LinkedIn feed.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

DPI = 200
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
OUT_DIR = os.path.join(HERE, "out")

INK = "#0B0E14"
SURFACE = "#12161F"
HAIRLINE = "#222A37"
TEXT = "#F3F5F9"
MUTED = "#8D97A8"
FAINT = "#5A6472"
GOLD = "#F5C44C"

REG, MED, SEMI, BOLD, XBOLD = 400, 500, 600, 700, 800


def register_fonts():
    for name in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
                 "Inter-Bold.ttf", "Inter-ExtraBold.ttf", "Inter-Black.ttf"]:
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["axes.unicode_minus"] = False


ANSWERS = [
    ("Q1", "uv version (setup check)", "0.12.12", "not graded"),
    ("Q2", "locked scikit-learn", "1.7.2", "uv sync --locked"),
    ("Q3", "load model, lead 1", "0.533", "raw 0.532994"),
    ("Q4", "serve model, lead 2", "0.770", "raw 0.769799"),
    ("Q5", "Python base image", "3.11.15-slim-bookworm", "Dockerfile"),
    ("Q6", "container, lead 2", "0.770", "same as Q4"),
]

EYEBROW = "ML ZOOMCAMP 2026 · HOMEWORK 5"
TITLE = "Deployment: las 6 respuestas"
SUBTITLE = "Lead scoring · FastAPI · Docker · 5,000 leads"


def make_card(idx, q, question, value, chip):
    fig = plt.figure(figsize=(1080 / DPI, 1350 / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1080)
    ax.set_ylim(0, 1350)
    ax.axis("off")
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)

    # eyebrow
    ax.text(90, 1260, EYEBROW, fontsize=20 * 72 / DPI, color=MUTED, weight=SEMI)
    # title
    ax.text(90, 1180, TITLE, fontsize=46 * 72 / DPI, color=TEXT, weight=XBOLD)
    ax.text(90, 1130, SUBTITLE, fontsize=26 * 72 / DPI, color=MUTED, weight=MED)

    # card panel
    from matplotlib.patches import FancyBboxPatch
    panel = FancyBboxPatch((90, 420), 900, 640, boxstyle="round,pad=0,rounding_size=24",
                           linewidth=1, edgecolor=HAIRLINE, facecolor=SURFACE)
    ax.add_patch(panel)

    ax.text(130, 980, f"{q}", fontsize=22 * 72 / DPI, color=GOLD, weight=BOLD)
    ax.text(130, 900, question, fontsize=30 * 72 / DPI, color=TEXT, weight=SEMI)
    ax.text(130, 760, value, fontsize=64 * 72 / DPI, color=GOLD, weight=XBOLD)
    ax.text(130, 690, chip, fontsize=22 * 72 / DPI, color=MUTED, weight=MED)

    # footer
    ax.text(90, 120, "05-deployment · FastAPI + Docker · scikit-learn 1.7.2",
            fontsize=18 * 72 / DPI, color=FAINT, weight=MED)
    ax.text(90, 80, "DataTalksClub/machine-learning-zoomcamp", fontsize=18 * 72 / DPI,
            color=FAINT, weight=MED)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"hw5-{q.lower()}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


def main():
    register_fonts()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    for i, (q, question, value, chip) in enumerate(ANSWERS):
        make_card(i, q, question, value, chip)
    # title card
    fig = plt.figure(figsize=(1080 / DPI, 1350 / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1080); ax.set_ylim(0, 1350); ax.axis("off")
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    ax.text(90, 1260, EYEBROW, fontsize=20 * 72 / DPI, color=MUTED, weight=SEMI)
    ax.text(90, 1100, "Las 6 respuestas", fontsize=72 * 72 / DPI, color=TEXT, weight=XBOLD)
    ax.text(90, 1020, "Deployment: FastAPI + Docker", fontsize=32 * 72 / DPI, color=MUTED, weight=MED)
    ax.text(90, 960, "Lead scoring · 5,000 leads · scikit-learn 1.7.2",
            fontsize=24 * 72 / DPI, color=FAINT, weight=MED)
    ax.text(90, 200, "marcosnieves · #mlzoomcamp", fontsize=20 * 72 / DPI, color=FAINT, weight=MED)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "hw5-title.png"), dpi=DPI)
    plt.close(fig)
    print("wrote title")
    if args.preview:
        print("preview: open out/*.png")


if __name__ == "__main__":
    main()
