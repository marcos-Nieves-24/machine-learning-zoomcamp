#!/usr/bin/env python3
"""Professional flow-diagram animation of the Homework 5 pipeline.

Scene A: inference (Lead -> Normalize -> Vectorize -> Logistic Regression -> Predict Probability -> API Response)
Scene B: deployment (pipeline.bin -> Load Model -> FastAPI -> POST /predict -> Prediction Pipeline -> JSON Response) + Docker container.

Output: out/hw5-flow-v2-1920x1080.mp4 (H.264, ~24s, 16:9).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
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
GREEN = "#4ADE80"

W, H = 1920, 1080
FPS = 30
DUR = 24
NFRAMES = DUR * FPS

# ---------- Scene A ----------
sceneA_stages = ["Lead Record", "Normalize", "Vectorize", "Logistic\nRegression",
                  "Predict\nProbability", "API Response"]
sceneA_bodies = [
    '{\n  "lead_source": "paid_ads",\n  "industry": "technology",\n  "employment_status": "employed",\n  "location": "north_america",\n  "number_of_courses_viewed": 2,\n  "annual_income": 79276.0,\n  "interaction_count": 4,\n  "lead_score": 0.41\n}',
    'normalize_record(lead)\n{\n  "lead_source": "paid_ads",\n  "annual_income": 79276.0,\n  "lead_score": 0.41,\n  "number_of_courses_viewed": 2,\n  "interaction_count": 4\n}',
    "DictVectorizer\n-> sparse vector (12 dims)\none-hot + numeric",
    "coef_ . x + intercept_\n(liblinear, C=1.0)\nprocessing...",
    "predict_proba\n-> 0.532994",
    '{\n  "conversion_probability": 0.533,\n  "conversion": true\n}',
]
sceneA_descs = [
    "Raw lead input with 8 observable features.",
    "Apply training-set medians for missing numerics; 'NA' for missing categoricals.",
    "Convert dict to a numeric feature vector (one-hot for categoricals).",
    "Linear model: score = coef . x + intercept.",
    "Sigmoid of the score -> probability of conversion.",
    "Final JSON returned to the client.",
]
sceneA_status = ["waiting", "processing", "processing", "processing", "processing", "completed"]

# ---------- Scene B ----------
sceneB_stages = ["pipeline.bin", "Load Model", "FastAPI", "POST /predict",
                  "Prediction\nPipeline", "JSON\nResponse"]
sceneB_bodies = [
    "frozen artifact\nSHA256: 1646bbdc...\nDictVectorizer + LogReg",
    "pickle.load(pipeline.bin)\nloaded once at startup",
    "uvicorn predict:app\nhost 0.0.0.0:9696",
    "HTTP request\nlead JSON body",
    "normalize -> vectorize\n-> predict_proba",
    '{\n  "conversion_probability": 0.770,\n  "conversion": true\n}',
]
sceneB_descs = [
    "Frozen model artifact on disk.",
    "Deserialize the pipeline into memory.",
    "ASGI app exposing /health and /predict.",
    "Client sends an HTTP POST with the lead.",
    "Run the same inference pipeline.",
    "Probability + boolean returned to client.",
]

nA = len(sceneA_stages)
nB = len(sceneB_stages)

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)

# grid
for gx in range(0, W + 1, 100):
    ax.plot([gx, gx], [0, H], color="#151A24", lw=0.4, zorder=0)
for gy in range(0, H + 1, 100):
    ax.plot([0, W], [gy, gy], color="#151A24", lw=0.4, zorder=0)

title = ax.text(W / 2, H - 50, "ML Deployment Flow", fontsize=42, color=TEXT,
                weight="bold", ha="center", zorder=5)
subtitle = ax.text(W / 2, H - 90, "", fontsize=20, color=MUTED, ha="center", zorder=5)

# layout helpers
def make_nodes(stages, y_box):
    box_w, box_h = 220, 90
    gap = 70
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    start_x = (W - total_w) / 2
    boxes, glows, labels, centers, arrows = [], [], [], [], []
    for i, name in enumerate(stages):
        x = start_x + i * (box_w + gap)
        cx = x + box_w / 2
        centers.append(cx)
        glow = Circle((cx, y_box), 60, color=GOLD, alpha=0, zorder=2)
        ax.add_patch(glow)
        glows.append(glow)
        box = FancyBboxPatch((x, y_box - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0,rounding_size=12",
                             linewidth=2, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=3)
        ax.add_patch(box)
        boxes.append(box)
        lab = ax.text(cx, y_box, name, fontsize=16, color=TEXT, weight="bold",
                      ha="center", va="center", zorder=4)
        labels.append(lab)
        if i < len(stages) - 1:
            arr_line, = ax.plot([], [], color=GOLD, lw=2.5, zorder=2, solid_capstyle="round")
            arr_head = ax.plot([], [], marker=">", color=GOLD, markersize=10, zorder=2, linestyle="None")[0]
            arrows.append((arr_line, arr_head, x + box_w, x + box_w + gap))
        else:
            arrows.append(None)
    return boxes, glows, labels, centers, arrows, start_x, box_w, gap

# Scene A nodes (upper area)
yA = H * 0.72
boxesA, glowsA, labelsA, centersA, arrowsA, sx, bw, gap = make_nodes(sceneA_stages, yA)

# Scene B nodes (upper area) - created hidden
yB = H * 0.72
boxesB, glowsB, labelsB, centersB, arrowsB, sx2, bw2, gap2 = make_nodes(sceneB_stages, yB)
for b in boxesB + glowsB:
    b.set_alpha(0)
for lab in labelsB:
    lab.set_alpha(0)
for arr in arrowsB:
    if arr:
        arr[0].set_alpha(0)
        arr[1].set_alpha(0)

# Docker container box (scene B) - enclosing
docker_x = sx2 - 30
docker_y = yB - 130
docker_w = (len(sceneB_stages) * bw2 + (len(sceneB_stages) - 1) * gap2) + 60
docker_h = 260
docker_box = FancyBboxPatch((docker_x, docker_y), docker_w, docker_h,
                           boxstyle="round,pad=0,rounding_size=16",
                           linewidth=2, edgecolor="#3B82F6", facecolor="none",
                           linestyle="--", zorder=1, alpha=0)
ax.add_patch(docker_box)
docker_label = ax.text(docker_x + 16, docker_y + docker_h - 24, "Docker container",
                      fontsize=16, color="#3B82F6", weight="bold", zorder=2, alpha=0)

# inspection panel (bottom)
panel_y = 120
panel_h = 230
panel = FancyBboxPatch((sx, panel_y), total_w if False else (W - 200), panel_h,
                       boxstyle="round,pad=0,rounding_size=14",
                       linewidth=1.5, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=2)
panel.set_x(100)
panel.set_width(W - 200)
ax.add_patch(panel)

panel_title = ax.text(130, panel_y + panel_h - 30, "Inspection panel",
                      fontsize=18, color=MUTED, weight="bold", zorder=3)
stage_name_t = ax.text(130, panel_y + panel_h - 62, "", fontsize=24, color=GOLD,
                       weight="bold", zorder=3)
desc_t = ax.text(130, panel_y + panel_h - 92, "", fontsize=16, color=MUTED, zorder=3)
# two columns: input (left), transform (mid), output (right)
col_w = (W - 200) / 3
input_t = ax.text(130, panel_y + panel_h - 125, "", fontsize=15, color=TEXT,
                  family="monospace", va="top", zorder=3)
transform_t = ax.text(130 + col_w, panel_y + panel_h - 125, "", fontsize=15,
                      color=TEXT, family="monospace", va="top", zorder=3)
output_t = ax.text(130 + 2 * col_w, panel_y + panel_h - 125, "", fontsize=15,
                   color=TEXT, family="monospace", va="top", zorder=3)
status_t = ax.text(W - 130, panel_y + panel_h - 30, "", fontsize=16, color=GREEN,
                   weight="bold", ha="right", zorder=3)

# probability gauge (only scene A)
gauge_x = 100
gauge_y = 60
gauge_w = W - 200
gauge_h = 24
gauge_bg = FancyBboxPatch((gauge_x, gauge_y), gauge_w, gauge_h,
                          boxstyle="round,pad=0,rounding_size=6",
                          linewidth=1, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=2)
ax.add_patch(gauge_bg)
gauge_fill = FancyBboxPatch((gauge_x, gauge_y), 0, gauge_h,
                            boxstyle="round,pad=0,rounding_size=6",
                            linewidth=0, facecolor=GOLD, zorder=3)
ax.add_patch(gauge_fill)
gauge_label = ax.text(gauge_x, gauge_y + gauge_h + 8, "conversion_probability",
                      fontsize=14, color=MUTED, zorder=3)
gauge_val = ax.text(gauge_x + gauge_w, gauge_y + gauge_h + 8, "0.000",
                    fontsize=18, color=GOLD, weight="bold", ha="right", zorder=3)

# traveling packet
packet = Circle((centersA[0], yA), 11, color=GOLD, zorder=6)
ax.add_patch(packet)
packet.set_alpha(0)

scene_label = ax.text(W / 2, H - 130, "", fontsize=18, color=MUTED, ha="center", zorder=5)

all_artists = ([title, subtitle, panel, panel_title, stage_name_t, desc_t, input_t,
                transform_t, output_t, status_t, gauge_bg, gauge_fill, gauge_label,
                gauge_val, packet, scene_label, docker_box, docker_label]
               + boxesA + labelsA + glowsA + boxesB + labelsB + glowsB)

# timing
SCENE_A_DUR = 12.0
SCENE_B_DUR = 12.0
TOTAL = SCENE_A_DUR + SCENE_B_DUR
segA = SCENE_A_DUR / (nA + 1)  # +1 for intro
segB = SCENE_B_DUR / (nB + 1)


def alpha_fade(t, start, length):
    if t < start:
        return 0.0
    if t > start + length:
        return 1.0
    return (t - start) / length


def update(frame):
    t = frame / FPS
    inA = t < SCENE_A_DUR

    # visibility
    for b, g, lab in zip(boxesA, glowsA, labelsA):
        b.set_alpha(1.0 if inA else 0.0)
        g.set_alpha(g.get_alpha() if inA else 0.0)
        lab.set_alpha(1.0 if inA else 0.0)
    for arr in arrowsA:
        if arr:
            arr[0].set_alpha(1.0 if inA else 0.0)
            arr[1].set_alpha(1.0 if inA else 0.0)
    for b, g, lab in zip(boxesB, glowsB, labelsB):
        b.set_alpha(0.0 if inA else 1.0)
        g.set_alpha(0.0)
        lab.set_alpha(0.0 if inA else 1.0)
    for arr in arrowsB:
        if arr:
            arr[0].set_alpha(0.0 if inA else 1.0)
            arr[1].set_alpha(0.0 if inA else 1.0)
    docker_box.set_alpha(0.0 if inA else 1.0)
    docker_label.set_alpha(0.0 if inA else 1.0)
    gauge_bg.set_alpha(1.0 if inA else 0.0)
    gauge_fill.set_alpha(1.0 if inA else 0.0)
    gauge_label.set_alpha(1.0 if inA else 0.0)
    gauge_val.set_alpha(1.0 if inA else 0.0)

    if inA:
        scene_label.set_text("Scene A - Inference")
        # stage index
        local_t = t
        intro = 1.0
        if local_t < intro:
            active = -1
            prog = local_t / intro
        else:
            idx = int((local_t - intro) / segA)
            active = min(nA - 1, idx)
            prog = ((local_t - intro) % segA) / segA
        subtitle.set_text("Inference pipeline")

        # highlight
        for i, (box, glow) in enumerate(zip(boxesA, glowsA)):
            if i == active:
                box.set_edgecolor(GOLD)
                box.set_linewidth(3)
                glow.set_alpha(0.25 + 0.1 * np.sin(frame * 0.25))
                glow.set_radius(60 + 5 * np.sin(frame * 0.25))
            elif i < active:
                box.set_edgecolor("#3A4455")
                box.set_linewidth(2)
                glow.set_alpha(0.0)
            else:
                box.set_edgecolor(HAIRLINE)
                box.set_linewidth(2)
                glow.set_alpha(0.0)

        # arrows
        for i, arr in enumerate(arrowsA):
            if arr is None:
                continue
            line, head, xa, xb = arr
            if i < active:
                line.set_data([xa, xb], [yA, yA])
                head.set_data([xb], [yA])
            elif i == active and active >= 0:
                if prog > 0.35:
                    p = min(1.0, (prog - 0.35) / 0.35)
                    xc = xa + (xb - xa) * p
                    line.set_data([xa, xc], [yA, yA])
                    if p >= 0.95:
                        head.set_data([xc], [yA])
                    else:
                        head.set_data([], [])
                else:
                    line.set_data([], [])
                    head.set_data([], [])
            else:
                line.set_data([], [])
                head.set_data([], [])

        # packet
        if 0 <= active < nA - 1:
            x0 = centersA[active]
            x1 = centersA[active + 1]
            if prog > 0.5:
                p = (prog - 0.5) / 0.5
                packet.set_center((x0 + (x1 - x0) * p, yA))
                packet.set_alpha(1.0)
            else:
                packet.set_center((x0, yA))
                packet.set_alpha(0.0)
        else:
            packet.set_center((centersA[-1], yA))
            packet.set_alpha(0.0)

        # inspection panel
        if active >= 0:
            stage_name_t.set_text(sceneA_stages[active])
            desc_t.set_text(sceneA_descs[active])
            input_t.set_text("Input:\n" + sceneA_bodies[max(0, active - 1)].split("\n")[0] if active > 0 else "Input:\nlead JSON")
            transform_t.set_text("Transform:\n" + sceneA_descs[active])
            output_t.set_text("Output:\n" + sceneA_bodies[active].split("\n")[0])
            status_t.set_text(sceneA_status[active].upper())
            status_t.set_color(GREEN if sceneA_status[active] == "completed" else GOLD)
        else:
            stage_name_t.set_text("Scene A - Inference")
            desc_t.set_text("Nodes appear, then connections are drawn.")
            input_t.set_text("")
            transform_t.set_text("")
            output_t.set_text("")
            status_t.set_text("WAITING")
            status_t.set_color(MUTED)

        # gauge
        if active == nA - 1:
            tt = min(1.0, prog / 0.7)
            val = 0.532994 * tt
            gauge_fill.set_width(min(gauge_w, val / 0.532994 * gauge_w))
            gauge_val.set_text(f"{val:.3f}")
        else:
            gauge_fill.set_width(0)
            gauge_val.set_text("0.000")

    else:
        scene_label.set_text("Scene B - Deployment")
        subtitle.set_text("Deployment: pipeline.bin -> FastAPI -> Docker")
        t2 = t - SCENE_A_DUR
        intro = 1.0
        if t2 < intro:
            active = -1
            prog = t2 / intro
        else:
            idx = int((t2 - intro) / segB)
            active = min(nB - 1, idx)
            prog = ((t2 - intro) % segB) / segB

        for i, (box, glow) in enumerate(zip(boxesB, glowsB)):
            if i == active:
                box.set_edgecolor(GOLD)
                box.set_linewidth(3)
                glow.set_alpha(0.25 + 0.1 * np.sin(frame * 0.25))
                glow.set_radius(60 + 5 * np.sin(frame * 0.25))
            elif i < active:
                box.set_edgecolor("#3A4455")
                box.set_linewidth(2)
                glow.set_alpha(0.0)
            else:
                box.set_edgecolor(HAIRLINE)
                box.set_linewidth(2)
                glow.set_alpha(0.0)

        for i, arr in enumerate(arrowsB):
            if arr is None:
                continue
            line, head, xa, xb = arr
            if i < active:
                line.set_data([xa, xb], [yB, yB])
                head.set_data([xb], [yB])
            elif i == active and active >= 0:
                if prog > 0.35:
                    p = min(1.0, (prog - 0.35) / 0.35)
                    xc = xa + (xb - xa) * p
                    line.set_data([xa, xc], [yB, yB])
                    if p >= 0.95:
                        head.set_data([xc], [yB])
                    else:
                        head.set_data([], [])
                else:
                    line.set_data([], [])
                    head.set_data([], [])
            else:
                line.set_data([], [])
                head.set_data([], [])

        if 0 <= active < nB - 1:
            x0 = centersB[active]
            x1 = centersB[active + 1]
            if prog > 0.5:
                p = (prog - 0.5) / 0.5
                packet.set_center((x0 + (x1 - x0) * p, yB))
                packet.set_alpha(1.0)
            else:
                packet.set_center((x0, yB))
                packet.set_alpha(0.0)
        else:
            packet.set_center((centersB[-1], yB))
            packet.set_alpha(0.0)

        if active >= 0:
            stage_name_t.set_text(sceneB_stages[active])
            desc_t.set_text(sceneB_descs[active])
            input_t.set_text("Input:\n" + sceneB_bodies[max(0, active - 1)].split("\n")[0] if active > 0 else "Input:\nartifact on disk")
            transform_t.set_text("Transform:\n" + sceneB_descs[active])
            output_t.set_text("Output:\n" + sceneB_bodies[active].split("\n")[0])
            status_t.set_text("COMPLETED" if active == nB - 1 else "PROCESSING")
            status_t.set_color(GREEN if active == nB - 1 else GOLD)
        else:
            stage_name_t.set_text("Scene B - Deployment")
            desc_t.set_text("Nodes appear, then connections are drawn.")
            input_t.set_text("")
            transform_t.set_text("")
            output_t.set_text("")
            status_t.set_text("WAITING")
            status_t.set_color(MUTED)

    return all_artists


anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "hw5-flow-v2-1920x1080.mp4")
writer = FFMpegWriter(fps=FPS, metadata=dict(title="HW5 flow v2"), bitrate=3500)
anim.save(out_path, writer=writer, dpi=100)
plt.close(fig)
print("wrote", out_path)
