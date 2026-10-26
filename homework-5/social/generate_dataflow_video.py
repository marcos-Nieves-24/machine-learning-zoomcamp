#!/usr/bin/env python3
"""Render an animated flow diagram of the Homework 5 model pipeline.

Nodes + animated arrows + a traveling data packet + live probability gauge.

Output: out/hw5-dataflow-1920x1080.mp4 (H.264, 14s, 16:9).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

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
GLOW = "#F5C44C"

W, H = 1920, 1080
FPS = 30
DUR = 14
NFRAMES = DUR * FPS

stages = ["1. Lead", "2. Normalize", "3. Vectorize", "4. Score", "5. proba", "6. Output"]
bodies = [
    '{\n  "lead_source": "paid_ads",\n  "annual_income": 79276,\n  "lead_score": 0.41\n}',
    '{\n  "lead_source": "paid_ads",\n  "annual_income": 79276.0,\n  "lead_score": 0.41\n}',
    "sparse vector\n(12 dims)",
    "coef_ . x + b\n(liblinear, C=1.0)",
    "P(converted=1)\n= 0.532994",
    '{"prob": 0.533,\n "conversion": true}',
]

n = len(stages)
box_w, box_h = 230, 100
gap = 90
total_w = n * box_w + (n - 1) * gap
start_x = (W - total_w) / 2
y_box = H * 0.62
seg = (DUR - 2) / n  # seconds per stage

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)

# subtle grid
for gx in range(0, W + 1, 80):
    ax.plot([gx, gx], [0, H], color="#1A1F2B", lw=0.5, zorder=0)
for gy in range(0, H + 1, 80):
    ax.plot([0, W], [gy, gy], color="#1A1F2B", lw=0.5, zorder=0)

title = ax.text(W / 2, H - 60, "Lead scoring - data flow", fontsize=46, color=TEXT,
                weight="bold", ha="center", zorder=3)
subtitle = ax.text(W / 2, H - 105, "Frozen pipeline: pickle.load  ->  predict_proba",
                   fontsize=22, color=MUTED, ha="center", zorder=3)

box_centers = []
boxes = []
labels = []
glows = []
for i, name in enumerate(stages):
    x = start_x + i * (box_w + gap)
    cx = x + box_w / 2
    box_centers.append(cx)
    glow = Circle((cx, y_box), 70, color=GOLD, alpha=0, zorder=2)
    ax.add_patch(glow)
    glows.append(glow)
    box = FancyBboxPatch((x, y_box - box_h / 2), box_w, box_h,
                         boxstyle="round,pad=0,rounding_size=14",
                         linewidth=2, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=3)
    ax.add_patch(box)
    boxes.append(box)
    lab = ax.text(cx, y_box + 16, name, fontsize=20, color=TEXT, weight="bold",
                  ha="center", zorder=4)
    labels.append(lab)
    # arrow
    if i < n - 1:
        ax.annotate("", xy=(x + box_w + 6, y_box),
                    xytext=(x + box_w + gap - 6, y_box),
                    arrowprops=dict(arrowstyle="->", color=HAIRLINE, lw=2.5, zorder=2))

# data panel
panel_y = 150
panel_h = 260
panel = FancyBboxPatch((start_x, panel_y), total_w, panel_h,
                       boxstyle="round,pad=0,rounding_size=14",
                       linewidth=1.5, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=2)
ax.add_patch(panel)
panel_title = ax.text(start_x + 24, panel_y + panel_h - 32, "Data at this stage",
                      fontsize=18, color=MUTED, weight="bold", zorder=3)
data_text = ax.text(start_x + 24, panel_y + panel_h - 70, "", fontsize=17,
                    color=TEXT, family="monospace", va="top", zorder=3)

# probability gauge
gauge_x = start_x
gauge_y = 70
gauge_w = total_w
gauge_h = 30
gauge_bg = FancyBboxPatch((gauge_x, gauge_y), gauge_w, gauge_h,
                          boxstyle="round,pad=0,rounding_size=8",
                          linewidth=1, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=2)
ax.add_patch(gauge_bg)
gauge_fill = FancyBboxPatch((gauge_x, gauge_y), 0, gauge_h,
                            boxstyle="round,pad=0,rounding_size=8",
                            linewidth=0, facecolor=GOLD, zorder=3)
ax.add_patch(gauge_fill)
gauge_label = ax.text(gauge_x, gauge_y + gauge_h + 10, "conversion_probability",
                      fontsize=15, color=MUTED, zorder=3)
gauge_val = ax.text(gauge_x + gauge_w, gauge_y + gauge_h + 10, "0.000",
                    fontsize=20, color=GOLD, weight="bold", ha="right", zorder=3)
conv_val = ax.text(gauge_x + gauge_w / 2, gauge_y - 26, "conversion: false",
                   fontsize=18, color=TEXT, ha="center", zorder=3)

# traveling packet
packet = Circle((box_centers[0], y_box), 12, color=GOLD, zorder=5)
ax.add_patch(packet)
packet.set_alpha(0)

all_artists = [title, subtitle, panel, panel_title, data_text, gauge_bg, gauge_fill,
               gauge_label, gauge_val, conv_val, packet] + boxes + labels + glows


def update(frame):
    t = frame / FPS
    # active stage
    active = min(n - 1, int(t / seg))
    local = (t - active * seg) / seg  # 0..1 within stage

    for i, (box, glow) in enumerate(zip(boxes, glows)):
        if i == active:
            box.set_edgecolor(GOLD)
            box.set_linewidth(3)
            glow.set_alpha(0.25 + 0.15 * np.sin(frame * 0.3))
            glow.set_radius(70 + 6 * np.sin(frame * 0.3))
        elif i < active:
            box.set_edgecolor("#3A4455")
            box.set_linewidth(2)
            glow.set_alpha(0.0)
        else:
            box.set_edgecolor(HAIRLINE)
            box.set_linewidth(2)
            glow.set_alpha(0.0)

    data_text.set_text(bodies[active])

    # packet travels from current box center to next
    if active < n - 1:
        x0 = box_centers[active]
        x1 = box_centers[active + 1]
        # move during second half of stage
        if local > 0.5:
            p = (local - 0.5) / 0.5
            px = x0 + (x1 - x0) * p
            packet.set_center((px, y_box))
            packet.set_alpha(1.0)
        else:
            packet.set_center((x0, y_box))
            packet.set_alpha(0.0)
    else:
        packet.set_center((box_centers[-1], y_box))
        packet.set_alpha(0.0)

    # gauge fills only at final stage
    if active == n - 1:
        tt = min(1.0, local / 0.6)
        val = 0.532994 * tt
        gauge_fill.set_width(min(gauge_w, val / 0.532994 * gauge_w))
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
writer = FFMpegWriter(fps=FPS, metadata=dict(title="HW5 data flow"), bitrate=3000)
anim.save(out_path, writer=writer, dpi=100)
plt.close(fig)
print("wrote", out_path)
