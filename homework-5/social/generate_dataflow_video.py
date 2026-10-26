#!/usr/bin/env python3
"""Render a data-flow animation of the Homework 5 model pipeline.

Horizontal pipeline: lead -> normalize -> vectorize -> score -> output,
with a moving highlight and a live probability gauge.

Output: out/hw5-dataflow-1920x1080.mp4 (H.264, 12s, 16:9).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIR = os.path.join(HERE, "fonts")

for name in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
             "Inter-Bold.ttf", "Inter-ExtraBold.ttf", "Inter-Black.ttf"]:
    p = os.path.join(FONT_DIR, name)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Inter"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0B0E14"
SURFACE = "#12161F"
HAIRLINE = "#222A37"
TEXT = "#F3F5F9"
MUTED = "#8D97A8"
GOLD = "#F5C44C"

W, H = 1920, 1080
FPS = 30
DUR = 12
NFRAMES = DUR * FPS

stages = [
    ("1. Lead", "raw JSON"),
    ("2. normalize", "medians"),
    ("3. Vectorize", "one-hot"),
    ("4. Score", "liblinear"),
    ("5. proba", "0.532994"),
    ("6. Output", "0.533"),
]

bodies = [
    '{\n  "lead_source": "paid_ads",\n  "annual_income": 79276,\n  "lead_score": 0.41\n}',
    '{\n  "lead_source": "paid_ads",\n  "annual_income": 79276.0,\n  "lead_score": 0.41\n}',
    "sparse vector\n(12 dims)",
    "coef_ · x + b",
    "P(converted=1)",
    '{"prob": 0.533,\n "conversion": true}',
]

n = len(stages)
box_w, box_h = 250, 110
gap = 60
total_w = n * box_w + (n - 1) * gap
start_x = (W - total_w) / 2
y_box = H * 0.58

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)

title = ax.text(W / 2, H - 70, "Lead scoring — data flow", fontsize=44, color=TEXT,
                weight="bold", ha="center")
subtitle = ax.text(W / 2, H - 115, "Frozen pipeline: pickle.load  ->  predict_proba",
                   fontsize=22, color=MUTED, ha="center")

boxes = []
labels = []
body_texts = []
for i, (name, chip) in enumerate(stages):
    x = start_x + i * (box_w + gap)
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x, y_box - box_h / 2), box_w, box_h,
                         boxstyle="round,pad=0,rounding_size=14",
                         linewidth=2, edgecolor=HAIRLINE, facecolor=SURFACE)
    ax.add_patch(box)
    lab = ax.text(x + box_w / 2, y_box + 18, name, fontsize=20, color=TEXT,
                  weight="bold", ha="center")
    chip_t = ax.text(x + box_w / 2, y_box - 18, chip, fontsize=14, color=MUTED,
                     ha="center")
    boxes.append(box)
    labels.append((lab, chip_t))
    # arrow
    if i < n - 1:
        ax.annotate("", xy=(x + box_w + 4, y_box),
                    xytext=(x + box_w + gap - 4, y_box),
                    arrowprops=dict(arrowstyle="->", color=HAIRLINE, lw=2.5))

# data panel below
panel_y = 180
panel_h = 260
panel = FancyBboxPatch((start_x, panel_y), total_w, panel_h,
                       boxstyle="round,pad=0,rounding_size=14",
                       linewidth=1.5, edgecolor=HAIRLINE, facecolor=SURFACE)
ax.add_patch(panel)
panel_title = ax.text(start_x + 20, panel_y + panel_h - 30, "Data at this stage",
                      fontsize=18, color=MUTED, weight="bold")
data_text = ax.text(start_x + 20, panel_y + panel_h - 70, "", fontsize=18,
                    color=TEXT, family="monospace", va="top")

# probability gauge
gauge_x = start_x
gauge_y = 90
gauge_w = total_w
gauge_h = 34
gauge_bg = FancyBboxPatch((gauge_x, gauge_y), gauge_w, gauge_h,
                          boxstyle="round,pad=0,rounding_size=8",
                          linewidth=1, edgecolor=HAIRLINE, facecolor=SURFACE)
ax.add_patch(gauge_bg)
gauge_fill = FancyBboxPatch((gauge_x, gauge_y), 0, gauge_h,
                            boxstyle="round,pad=0,rounding_size=8",
                            linewidth=0, facecolor=GOLD)
ax.add_patch(gauge_fill)
gauge_label = ax.text(gauge_x, gauge_y + gauge_h + 12, "conversion_probability",
                      fontsize=16, color=MUTED)
gauge_val = ax.text(gauge_x + gauge_w, gauge_y + gauge_h + 12, "0.000",
                    fontsize=20, color=GOLD, weight="bold", ha="right")
conv_val = ax.text(gauge_x + gauge_w / 2, gauge_y - 30, "conversion: false",
                   fontsize=20, color=TEXT, ha="center")

# moving data packet
packet = Circle((start_x - 40, y_box), 10, color=GOLD, zorder=5)
ax.add_patch(packet)
packet.set_alpha(0)

all_artists = [title, subtitle, panel, panel_title, data_text, gauge_bg, gauge_fill,
               gauge_label, gauge_val, conv_val, packet] + [b for trio in [] for b in trio]
for box in boxes:
    all_artists.append(box)
for lab, chip_t in labels:
    all_artists += [lab, chip_t]


def update(frame):
    # stage index active: 0..5 over frames, each ~1.8s
    seg = DUR / n  # ~2s
    active = min(n - 1, int(frame / (seg * FPS)))
    # highlight active box
    for i, box in enumerate(boxes):
        if i == active:
            box.set_edgecolor(GOLD)
            box.set_linewidth(3)
            box.set_facecolor("#1A1F2B")
        elif i < active:
            box.set_edgecolor("#3A4455")
            box.set_facecolor(SURFACE)
        else:
            box.set_edgecolor(HAIRLINE)
            box.set_facecolor(SURFACE)
    # data panel
    data_text.set_text(bodies[active])
    # packet position: move along top of active box
    x = start_x + active * (box_w + gap) + box_w / 2
    packet.set_center((x, y_box + box_h / 2 + 18))
    packet.set_alpha(1.0)
    # gauge fills only at final stage
    if active == n - 1:
        t = min(1.0, (frame - active * seg * FPS) / (1.2 * FPS))
        val = 0.532994 * t
        gauge_fill.set_width(min(gauge_w, val * gauge_w / 0.532994))
        gauge_val.set_text(f"{val:.3f}")
        conv_val.set_text("conversion: true" if val >= 0.5 else "conversion: false")
    else:
        gauge_fill.set_width(0)
        gauge_val.set_text("0.000")
        conv_val.set_text("conversion: false")
    return all_artists


anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "hw5-dataflow-1920x1080.mp4")
writer = FFMpegWriter(fps=FPS, metadata=dict(title="HW5 data flow"), bitrate=2500)
anim.save(out_path, writer=writer, dpi=100)
plt.close(fig)
print("wrote", out_path)
