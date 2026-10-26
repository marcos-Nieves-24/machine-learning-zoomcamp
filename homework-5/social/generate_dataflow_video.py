#!/usr/bin/env python3
"""Render a data-flow animation of the Homework 5 model pipeline.

Shows: lead (JSON) -> normalize_record -> DictVectorizer -> LogisticRegression
       -> predict_proba -> {conversion_probability, conversion}

Output: out/hw5-dataflow-1920x1080.mp4 (H.264, 15s, 16:9).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIR = os.path.join(HERE, "fonts")

for name in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
             "Inter-Bold.ttf", "Inter-ExtraBold.ttf"]:
    p = os.path.join(FONT_DIR, name)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Inter"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0B0E14"
TEXT = "#F3F5F9"
MUTED = "#8D97A8"
GOLD = "#F5C44C"
HAIRLINE = "#222A37"

W, H = 1920, 1080
FPS = 30
DUR = 15  # seconds
NFRAMES = DUR * FPS

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}
record = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

steps = [
    ("1. Input lead (raw)", json.dumps(lead, indent=2)),
    ("2. normalize_record", json.dumps(record, indent=2)),
    ("3. DictVectorizer", "one-hot + numeric -> sparse vector (12 dims)"),
    ("4. LogisticRegression", "coef_ · x + intercept_  (liblinear, C=1.0)"),
    ("5. predict_proba", "P(converted=1) = 0.532994"),
    ("6. Output", '{"conversion_probability": 0.533, "conversion": true}'),
]


def alpha_for(frame, start, length):
    t = (frame - start) / max(length, 1)
    if t < 0:
        return 0.0
    if t > 1:
        return 1.0
    return t


fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)

title = ax.text(W / 2, H - 80, "Lead scoring — data flow", fontsize=40, color=TEXT,
                weight="bold", ha="center")
subtitle = ax.text(W / 2, H - 130, "Frozen pipeline: pickle.load -> predict_proba",
                   fontsize=22, color=MUTED, ha="center")

# boxes positions
box_w, box_h = 520, 300
x_left = 120
x_right = W - box_w - 120
y_top = H - 260
gap = 60

texts = []
for i, (name, body) in enumerate(steps):
    y = y_top - i * (box_h + gap)
    if y < 60:
        break
    # box
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x_left, y - box_h), box_w, box_h,
                         boxstyle="round,pad=0,rounding_size=16",
                         linewidth=1.5, edgecolor=HAIRLINE, facecolor="#12161F")
    ax.add_patch(box)
    t_name = ax.text(x_left + 24, y - 36, name, fontsize=24, color=GOLD, weight="bold")
    t_body = ax.text(x_left + 24, y - 70, body, fontsize=16, color=TEXT,
                     family="monospace", va="top")
    t_name.set_alpha(0)
    t_body.set_alpha(0)
    box.set_alpha(0)
    texts.append((box, t_name, t_body))
    # arrow to next
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_left + box_w / 2, y - box_h - 8),
                    xytext=(x_left + box_w / 2, y - box_h + 8),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))

# right side: probability bar
bar_x = x_right
bar_y = H / 2
bar = ax.barh([0], [0], height=60, color=GOLD, left=bar_x)[0]
ax.text(bar_x, bar_y + 60, "conversion_probability", fontsize=20, color=MUTED)
prob_text = ax.text(bar_x + 20, bar_y - 10, "0.000", fontsize=48, color=GOLD, weight="bold")

# final conversion
conv_text = ax.text(bar_x, bar_y - 100, "conversion: false", fontsize=24, color=TEXT)

all_artists = [title, subtitle] + [a for trio in texts for a in trio] + [bar, prob_text, conv_text]


def update(frame):
    # reveal steps progressively: each step over ~1s, staggered
    for i, (box, t_name, t_body) in enumerate(texts):
        start = i * (FPS * 1.6) + FPS * 0.5
        a = alpha_for(frame, start, FPS * 0.8)
        box.set_alpha(a * 0.9)
        t_name.set_alpha(a)
        t_body.set_alpha(a)
    # animate probability bar after step 5 revealed
    step5_start = 5 * (FPS * 1.6) + FPS * 0.5
    if frame >= step5_start:
        t = min(1.0, (frame - step5_start) / (FPS * 1.2))
        val = 0.532994 * t
        bar.set_width(val * 400)
        prob_text.set_text(f"{val:.3f}")
        conv_text.set_text("conversion: true" if val >= 0.5 else "conversion: false")
    return all_artists


anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "hw5-dataflow-1920x1080.mp4")
writer = FFMpegWriter(fps=FPS, metadata=dict(title="HW5 data flow"), bitrate=2000)
anim.save(out_path, writer=writer)
plt.close(fig)
print("wrote", out_path)
