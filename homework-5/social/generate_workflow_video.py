#!/usr/bin/env python3
"""Workflow-Builder style animation, faithful to the reference image.

Reference: dark canvas, faint grid + circuit traces, 4 group containers,
neon node cards with ports, bezier connections with glow + traveling packets.

Scene A (inference, lead-1 -> 0.533):
  DATA INPUT [Lead Record] -> DATA PREPARATION [Normalize, Vectorize]
  -> INFERENCE [Logistic Regression, Predict Probability] -> OUTPUT [API Response]

Scene B (deployment, lead-2 -> 0.770):
  ARTIFACT [pipeline.bin] -> SERVICE [Load Model, FastAPI]
  -> REQUEST [POST /predict, Prediction Pipeline] -> OUTPUT [JSON Response]
  Docker dashed container encloses SERVICE + REQUEST.

Real values from homework-5.ipynb. No invented services.
Output: out/hw5-workflow-1920x1080.mp4 (H.264, 26s, 16:9).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIR = os.path.join(HERE, "fonts")

for _n in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
           "Inter-Bold.ttf", "Inter-ExtraBold.ttf", "Inter-Black.ttf"]:
    _p = os.path.join(FONT_DIR, _n)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = "Inter"
plt.rcParams["axes.unicode_minus"] = False

# --- palette (from reference) ---
INK = "#0A0E14"
SURFACE = "#12161F"
GROUP_FILL = "#10141D"
HAIRLINE = "#222A37"
TEXT = "#F3F5F9"
MUTED = "#8D97A8"
FAINT = "#5A6472"
CYAN = "#22D3EE"    # DATA INPUT edges + nodes
GREEN = "#4ADE80"   # PREPARATION edges + nodes
BLUE = "#60A5FA"    # INFERENCE edges + nodes
AMBER = "#FBBF24"   # OUTPUT edges + nodes
DOCKER = "#3B82F6"

W, H = 1920, 1080
FPS = 30
SCENE_DUR = 13.5
DUR = SCENE_DUR * 2
NFRAMES = int(DUR * FPS)

# --- real stage data (verified against notebook) ---
# Scene A uses lead-1 (paid_ads, 79276.0, 0.41) -> 0.532994 -> 0.533
# Scene B uses lead-2 (organic_search, 80304.0, 0.74) -> 0.769799 -> 0.770


def make_scene_a():
    groups = [
        {"title": "DATA INPUT", "color": CYAN},
        {"title": "DATA PREPARATION", "color": GREEN},
        {"title": "INFERENCE", "color": BLUE},
        {"title": "OUTPUT", "color": AMBER},
    ]
    nodes = [
        {"title": "Lead Record", "icon": "=", "group": 0, "color": CYAN,
         "detail": "paid_ads · tech\n8 fields",
         "desc": "Raw lead input with 8 observable features.",
         "inp": "lead JSON (raw)",
         "out": "dict with 8 fields"},
        {"title": "Normalize", "icon": "@", "group": 1, "color": GREEN,
         "detail": "normalize_record()\nmedians · NA",
         "desc": "Apply training-set medians; 'NA' for missing categoricals.",
         "inp": "raw dict",
         "out": "clean dict"},
        {"title": "Vectorize", "icon": "fx", "group": 1, "color": GREEN,
         "detail": "DictVectorizer\n12-dims vector",
         "desc": "Dict to numeric vector (one-hot + numeric).",
         "inp": "clean dict",
         "out": "12-dim vector"},
        {"title": "Logistic Regression", "icon": "f", "group": 2, "color": BLUE,
         "detail": "liblinear · C=1.0\ncoef·x + b",
         "desc": "Linear score from the frozen pipeline.",
         "inp": "12-dim vector",
         "out": "logit score"},
        {"title": "Predict Probability", "icon": "%", "group": 2, "color": BLUE,
         "detail": "predict_proba()\n0.532994",
         "desc": "Sigmoid of the score: estimated P(converted=1).",
         "inp": "logit score",
         "out": "0.532994"},
        {"title": "API Response", "icon": ">>", "group": 3, "color": AMBER,
         "detail": "0.533 · true\nJSON",
         "desc": "Final JSON returned to the client.",
         "inp": "0.532994",
         "out": '{"prob": 0.533}'},
    ]
    edges = [(0, 1, CYAN), (1, 2, GREEN), (2, 3, GREEN), (3, 4, BLUE), (4, 5, AMBER)]
    return groups, nodes, edges


def make_scene_b():
    groups = [
        {"title": "ARTIFACT", "color": CYAN},
        {"title": "SERVICE", "color": GREEN},
        {"title": "REQUEST", "color": BLUE},
        {"title": "OUTPUT", "color": AMBER},
    ]
    nodes = [
        {"title": "pipeline.bin", "icon": "#", "group": 0, "color": CYAN,
         "detail": "frozen · 1.9 KB\nsha 1646bbdc",
         "desc": "Frozen artifact on disk (DictVectorizer + LogReg).",
         "inp": "file on disk",
         "out": "bytes"},
        {"title": "Load Model", "icon": "@", "group": 1, "color": GREEN,
         "detail": "pickle.load()\nonce at startup",
         "desc": "Deserialize the pipeline into memory.",
         "inp": "pipeline.bin",
         "out": "pipeline object"},
        {"title": "FastAPI", "icon": "S", "group": 1, "color": GREEN,
         "detail": "uvicorn :9696\n/health · /predict",
         "desc": "ASGI app exposing /health and /predict.",
         "inp": "pipeline object",
         "out": "running app"},
        {"title": "POST /predict", "icon": ">", "group": 2, "color": BLUE,
         "detail": "organic_search\n80304 · 0.74",
         "desc": "Client sends HTTP POST with lead-2 JSON.",
         "inp": "lead-2 JSON",
         "out": "request"},
        {"title": "Prediction Pipeline", "icon": "f", "group": 2, "color": BLUE,
         "detail": "normalize →\nvectorize → proba",
         "desc": "Same inference pipeline as Scene A.",
         "inp": "request",
         "out": "0.769799"},
        {"title": "JSON Response", "icon": ">>", "group": 3, "color": AMBER,
         "detail": "0.770 · true\nJSON",
         "desc": "Probability + boolean back to the client.",
         "inp": "0.769799",
         "out": '{"prob": 0.770}'},
    ]
    edges = [(0, 1, CYAN), (1, 2, GREEN), (2, 3, GREEN), (3, 4, BLUE), (4, 5, AMBER)]
    return groups, nodes, edges


GROUPS_A, NODES_A, EDGES_A = make_scene_a()
GROUPS_B, NODES_B, EDGES_B = make_scene_b()


# --- layout: 4 group boxes, nodes stacked inside ---
def layout(groups, nodes):
    usable = W - 120
    gaps = [40] * (len(groups) - 1)
    widths = [310, 390, 700, 280]
    # scale widths to fit
    total_gaps = sum(gaps)
    scale = (usable - total_gaps) / sum(widths)
    widths = [w * scale for w in widths]
    xs = []
    x = 60.0
    for i, w in enumerate(widths):
        xs.append((x, w))
        x += w + (gaps[i] if i < len(gaps) else 0)
    gy0, gy1 = 360.0, 930.0
    group_boxes = [(gx, gy0, gw, gy1 - gy0) for (gx, gw) in xs]
    # nodes per group
    from collections import defaultdict
    by_group = defaultdict(list)
    for idx, nd in enumerate(nodes):
        by_group[nd["group"]].append(idx)
    node_rects = [None] * len(nodes)
    for g, idxs in by_group.items():
        gx, gw = xs[g]
        cx = gx + gw / 2
        nw = min(250.0, gw - 44)
        nh = 150.0 if len(idxs) > 1 else 170.0
        if len(idxs) == 1:
            ys = [(gy0 + gy1) / 2]
        elif len(idxs) == 2:
            ys = [(gy0 + gy1) / 2 + 105, (gy0 + gy1) / 2 - 105]
        else:
            span = (gy1 - gy0) - 140
            ys = [gy1 - 90 - i * (span / max(1, len(idxs) - 1)) for i in range(len(idxs))]
        for ni, cy in zip(idxs, ys):
            node_rects[ni] = (cx - nw / 2, cy - nh / 2, nw, nh, cx, cy)
    return group_boxes, node_rects


GB_A, NR_A = layout(GROUPS_A, NODES_A)
GB_B, NR_B = layout(GROUPS_B, NODES_B)

# --- figure ---
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)

# faint grid
for gx in range(0, W + 1, 80):
    ax.plot([gx, gx], [0, H], color="#141A26", lw=0.5, zorder=0)
for gy in range(0, H + 1, 80):
    ax.plot([0, W], [gy, gy], color="#141A26", lw=0.5, zorder=0)
# faint circuit traces (corners, like reference)
for (cx0, cy0, segs) in [
    (1500, 930, [(0, 0), (60, 0), (60, 40), (120, 40)]),
    (60, 200, [(0, 0), (80, 0), (80, -50), (160, -50)]),
    (1560, 180, [(0, 0), (0, 50), (70, 50), (70, 110)]),
]:
    px, py = cx0, cy0
    for dx, dy in segs[1:]:
        ax.plot([px, cx0 + dx], [py, cy0 + dy], color="#1B2436", lw=1.2, zorder=0)
        px, py = cx0 + dx, cy0 + dy
    ax.plot([px], [py], marker="o", color="#1B2436", markersize=5, zorder=0)

title = ax.text(W / 2, H - 48, "Lead scoring — deployment flow", fontsize=40,
                color=TEXT, weight="bold", ha="center", zorder=20)
scene_tag = ax.text(W / 2, H - 88, "", fontsize=19, color=MUTED, ha="center", zorder=20)


def build_scene_artists(groups, nodes, edges, group_boxes, node_rects):
    g_patches, g_titles = [], []
    for (gx, gy0, gw, gh), g in zip(group_boxes, groups):
        p = FancyBboxPatch((gx, gy0), gw, gh,
                           boxstyle="round,pad=0,rounding_size=18",
                           linewidth=1.2, edgecolor="#2A3448", facecolor=GROUP_FILL,
                           zorder=1)
        ax.add_patch(p)
        g_patches.append(p)
        t = ax.text(gx + gw / 2, gy0 + gh - 26, g["title"], fontsize=15, color=MUTED,
                    weight="bold", ha="center", zorder=4)
        g_titles.append(t)
    n_glow, n_box, n_title, n_icon, n_detail, n_ports = [], [], [], [], [], []
    for (nx, ny, nw, nh, cx, cy), nd in zip(node_rects, nodes):
        gl = FancyBboxPatch((nx - 5, ny - 5), nw + 10, nh + 10,
                            boxstyle="round,pad=0,rounding_size=16",
                            linewidth=7, edgecolor=nd["color"], facecolor="none",
                            zorder=2)
        gl.set_alpha(0.0)
        ax.add_patch(gl)
        n_glow.append(gl)
        bx = FancyBboxPatch((nx, ny), nw, nh,
                            boxstyle="round,pad=0,rounding_size=14",
                            linewidth=2, edgecolor=HAIRLINE, facecolor=SURFACE,
                            zorder=3)
        ax.add_patch(bx)
        n_box.append(bx)
        n_title.append(ax.text(cx, ny + nh - 20, nd["title"], fontsize=13.5, color=TEXT,
                               weight="bold", ha="center", va="top", zorder=5))
        n_icon.append(ax.text(cx, cy + 14, nd["icon"], fontsize=28, color=nd["color"],
                              ha="center", va="center", zorder=5))
        n_detail.append(ax.text(cx, ny + 26, nd["detail"], fontsize=11, color=MUTED,
                                ha="center", va="bottom", zorder=5, family="monospace"))
        # ports: left/right + top/bottom (vertical edges use top/bottom)
        pl = Circle((nx, cy), 6, color=nd["color"], zorder=6)
        pr = Circle((nx + nw, cy), 6, color=nd["color"], zorder=6)
        pt = Circle((cx, ny + nh), 6, color=nd["color"], zorder=6)
        pb = Circle((cx, ny), 6, color=nd["color"], zorder=6)
        for _c in (pl, pr, pt, pb):
            ax.add_patch(_c)
        n_ports.append((pl, pr, pt, pb))
    # edges: glow + core + head + ports on curve ends
    e_glow, e_core, e_head = [], [], []
    for (a, b, col) in edges:
        gl, = ax.plot([], [], color=col, lw=8, alpha=0.16, zorder=2, solid_capstyle="round")
        co, = ax.plot([], [], color=col, lw=2.6, zorder=3, solid_capstyle="round")
        hd = ax.plot([], [], marker="o", color=col, markersize=7, zorder=4, linestyle="None")[0]
        e_glow.append(gl)
        e_core.append(co)
        e_head.append(hd)
    return (g_patches, g_titles, n_glow, n_box, n_title, n_icon, n_detail,
            n_ports, e_glow, e_core, e_head)


A = build_scene_artists(GROUPS_A, NODES_A, EDGES_A, GB_A, NR_A)
B = build_scene_artists(GROUPS_B, NODES_B, EDGES_B, GB_B, NR_B)

# Docker dashed container for scene B (encloses SERVICE + REQUEST groups)
dgx0 = GB_B[1][0] - 18
dgy0 = GB_B[1][1] - 26
dgx1 = GB_B[2][0] + GB_B[2][2] + 18
dgy1 = GB_B[1][1] + GB_B[1][3] + 26
docker_box = FancyBboxPatch((dgx0, dgy0), dgx1 - dgx0, dgy1 - dgy0,
                            boxstyle="round,pad=0,rounding_size=20",
                            linewidth=2, edgecolor=DOCKER, facecolor="none",
                            linestyle=(0, (8, 6)), zorder=1)
ax.add_patch(docker_box)
docker_label = ax.text(dgx0 + 16, dgy1 - 22, "Docker · zoomcamp-model:2026-hw5",
                       fontsize=15, color=DOCKER, weight="bold", zorder=5)

# --- inspection bar (slim, bottom) ---
PX, PW, PY, PH = 60, W - 120, 52, 250
panel = FancyBboxPatch((PX, PY), PW, PH,
                       boxstyle="round,pad=0,rounding_size=14",
                       linewidth=1.2, edgecolor=HAIRLINE, facecolor=SURFACE, zorder=5)
ax.add_patch(panel)
p_stage = ax.text(PX + 24, PY + PH - 30, "", fontsize=22, color="#F5C44C",
                  weight="bold", zorder=6)
p_desc = ax.text(PX + 24, PY + PH - 58, "", fontsize=14, color=MUTED, zorder=6)
p_io = ax.text(PX + 24, PY + 66, "", fontsize=13.5, color=TEXT, zorder=6,
               family="monospace", va="top")
p_status = ax.text(PX + PW - 24, PY + PH - 30, "", fontsize=15, color=TEXT,
                   weight="bold", ha="right", zorder=6)
# gauge (scene A + B final)
gau_x, gau_y, gau_w, gau_h = PX + PW * 0.55, PY + 26, PW * 0.42, 20
gau_bg = FancyBboxPatch((gau_x, gau_y), gau_w, gau_h,
                        boxstyle="round,pad=0,rounding_size=6",
                        linewidth=1, edgecolor=HAIRLINE, facecolor=INK, zorder=6)
ax.add_patch(gau_bg)
gau_fill = FancyBboxPatch((gau_x, gau_y), 0, gau_h,
                          boxstyle="round,pad=0,rounding_size=6",
                          linewidth=0, facecolor="#F5C44C", zorder=7)
ax.add_patch(gau_fill)
gau_txt = ax.text(gau_x + gau_w, gau_y + gau_h + 8, "", fontsize=16,
                  color="#F5C44C", weight="bold", ha="right", zorder=6)

packet_glow = Circle((0, 0), 16, color="#F5C44C", alpha=0.25, zorder=8)
packet = Circle((0, 0), 8, color="white", zorder=9)
packet_core = Circle((0, 0), 5, color="#F5C44C", zorder=10)
for c in (packet_glow, packet, packet_core):
    ax.add_patch(c)
    c.set_alpha(0)


def bezier(p0, p3, npts=80, vertical=False):
    x0, y0 = p0
    x1, y1 = p3
    if vertical:
        ts = np.linspace(0, 1, npts)
        return np.full_like(ts, x0, dtype=float), (1 - ts) * y0 + ts * y1
    dx = min(150.0, max(50.0, abs(x1 - x0) * 0.45))
    c1, c2 = (x0 + dx, y0), (x1 - dx, y1)
    ts = np.linspace(0, 1, npts)
    xs = (1 - ts) ** 3 * x0 + 3 * (1 - ts) ** 2 * ts * c1[0] + 3 * (1 - ts) * ts ** 2 * c2[0] + ts ** 3 * x1
    ys = (1 - ts) ** 3 * y0 + 3 * (1 - ts) ** 2 * ts * c1[1] + 3 * (1 - ts) * ts ** 2 * c2[1] + ts ** 3 * y1
    return xs, ys


def edge_points(nr_a, nr_b):
    ax0, ay0, aw, ah, acx, ay = nr_a
    bx0, by0, bw, bh, bcx, by = nr_b
    if abs(acx - bcx) < 1.0:
        # stacked nodes in one group: bottom of upper -> top of lower
        return (acx, ay0), (bcx, by0 + bh), True
    return (ax0 + aw, ay), (bx0, by), False


def set_scene_visible(artists, node_rects, visible):
    (g_p, g_t, n_gl, n_bx, n_ti, n_ic, n_de, n_po, e_gl, e_co, e_he) = artists
    a = 1.0 if visible else 0.0
    for p in g_p:
        p.set_alpha(a)
    for t in g_t + n_ti + n_ic + n_de:
        t.set_alpha(a)
    for b in n_bx:
        b.set_alpha(a)
    for pl, pr, pt, pb in n_po:
        pl.set_alpha(a)
        pr.set_alpha(a)
        pt.set_alpha(a)
        pb.set_alpha(a)
    for l in e_gl:
        l.set_alpha(0.16 * a if a else 0.0)
    for l in e_co + e_he:
        l.set_alpha(a)


INTRO, NDRAW, FLOW = 1.2, 1.1, 1.4  # seconds per phase


def render_scene(t, artists, groups, nodes, edges, node_rects, final_prob, scene_name):
    (g_p, g_t, n_gl, n_bx, n_ti, n_ic, n_de, n_po, e_gl, e_co, e_he) = artists
    n = len(nodes)
    t_nodes_end = INTRO + n * 0.28
    t_edges_end = t_nodes_end + NDRAW
    # groups
    ga = min(1.0, t / 0.9)
    for p in g_p:
        p.set_alpha(ga)
    for tt in g_t:
        tt.set_alpha(ga)
    # nodes stagger
    for i in range(n):
        na = np.clip((t - 0.5 - i * 0.28) / 0.5, 0, 1)
        n_bx[i].set_alpha(na)
        n_ti[i].set_alpha(na)
        n_ic[i].set_alpha(na)
        n_de[i].set_alpha(na)
        for _pp in n_po[i]:
            _pp.set_alpha(na)
        n_gl[i].set_alpha(0.0)
    # edges draw
    drawn = []
    for i, (a, b, col) in enumerate(edges):
        st = t_nodes_end + (i / max(1, len(edges))) * NDRAW
        frac = np.clip((t - st) / (NDRAW / max(1, len(edges)) * 1.6 + 0.25), 0, 1)
        (x0, y0), (x1, y1), vert = edge_points(node_rects[a], node_rects[b])
        xs, ys = bezier((x0, y0), (x1, y1), vertical=vert)
        k = max(2, int(len(xs) * frac)) if frac > 0 else 0
        if k > 0:
            e_gl[i].set_data(xs[:k], ys[:k])
            e_co[i].set_data(xs[:k], ys[:k])
            e_gl[i].set_alpha(0.16)
            e_co[i].set_alpha(1.0)
            if frac >= 0.99:
                e_he[i].set_data([xs[-1]], [ys[-1]])
                e_he[i].set_alpha(1.0)
            else:
                e_he[i].set_data([], [])
        else:
            e_gl[i].set_data([], [])
            e_co[i].set_data([], [])
            e_he[i].set_data([], [])
        drawn.append((xs, ys, frac))
    # flow
    active = -1
    pack_pos = None
    if t >= t_edges_end:
        ft = t - t_edges_end
        active = min(n - 1, int(ft / FLOW))
        local = (ft - active * FLOW) / FLOW
        # node states
        for i in range(n):
            if i == active:
                n_bx[i].set_edgecolor(nodes[i]["color"])
                n_bx[i].set_linewidth(3)
                n_gl[i].set_alpha(0.22 + 0.10 * np.sin(t * FPS * 0.25))
            elif i < active:
                n_bx[i].set_edgecolor(nodes[i]["color"])
                n_bx[i].set_linewidth(2)
                n_gl[i].set_alpha(0.0)
            else:
                n_bx[i].set_edgecolor(HAIRLINE)
                n_bx[i].set_linewidth(2)
                n_gl[i].set_alpha(0.0)
        # packet on incoming edge
        if active > 0 and local < 0.85:
            xs, ys, _ = drawn[active - 1]
            p = np.clip(local / 0.85, 0, 1)
            idx = min(len(xs) - 1, int(p * (len(xs) - 1)))
            pack_pos = (xs[idx], ys[idx])
        elif active < n - 1 and local >= 0.85:
            xs, ys, _ = drawn[active]
            p = np.clip((local - 0.85) / 0.15, 0, 1)
            idx = min(len(xs) - 1, int(p * (len(xs) - 1)))
            pack_pos = (xs[idx], ys[idx])
    else:
        for i in range(n):
            n_bx[i].set_edgecolor(HAIRLINE)
            n_bx[i].set_linewidth(2)
            n_gl[i].set_alpha(0.0)
    # panel
    if active >= 0:
        nd = nodes[active]
        p_stage.set_text(f"{active + 1}/{n} · {nd['title']}")
        p_stage.set_color(nd["color"])
        p_desc.set_text(nd["desc"])
        p_io.set_text(f"in:  {nd['inp']}\nout: {nd['out']}")
        done = active == n - 1 and local > 0.7
        p_status.set_text("COMPLETED" if done else "PROCESSING")
        p_status.set_color("#4ADE80" if done else "#F5C44C")
        # gauge fills on last two stages
        if active >= n - 2:
            base = 0.0 if active == n - 2 else 0.6
            vv = min(1.0, base + local * 0.5)
            gau_fill.set_width(gau_w * min(1.0, vv) * (final_prob / max(final_prob, 1e-9)) * final_prob / final_prob)
            # width proportional to prob value (0..1)
            gau_fill.set_width(gau_w * final_prob * min(1.0, vv + 0.15))
            gau_txt.set_text(f"{final_prob * min(1.0, vv + 0.15):.3f}")
        else:
            gau_fill.set_width(0)
            gau_txt.set_text("0.000")
    else:
        p_stage.set_text(scene_name)
        p_stage.set_color(TEXT)
        p_desc.set_text("Nodes appear, connections are drawn, then data flows.")
        p_io.set_text("")
        p_status.set_text("WAITING")
        p_status.set_color(MUTED)
        gau_fill.set_width(0)
        gau_txt.set_text("0.000")
    return pack_pos


def update(frame):
    t = frame / FPS
    in_a = t < SCENE_DUR
    set_scene_visible(A, NR_A, in_a)
    set_scene_visible(B, NR_B, not in_a)
    docker_box.set_alpha(0.0 if in_a else 1.0)
    docker_label.set_alpha(0.0 if in_a else 1.0)
    if in_a:
        scene_tag.set_text("Scene A · Inference  —  lead-1 → 0.533")
        pp = render_scene(t, A, GROUPS_A, NODES_A, EDGES_A, NR_A, 0.532994,
                          "Scene A · Inference")
    else:
        scene_tag.set_text("Scene B · Deployment  —  lead-2 → 0.770 · Docker")
        pp = render_scene(t - SCENE_DUR, B, GROUPS_B, NODES_B, EDGES_B, NR_B, 0.769799,
                          "Scene B · Deployment")
    if pp is None:
        for c in (packet_glow, packet, packet_core):
            c.set_alpha(0)
    else:
        for c, a, r in ((packet_glow, 0.25, 16), (packet, 1.0, 8), (packet_core, 1.0, 5)):
            c.center = pp
            c.set_radius(r)
            c.set_alpha(a)
    return [title, scene_tag]


anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "hw5-workflow-1920x1080.mp4")
writer = FFMpegWriter(fps=FPS, metadata={"title": "HW5 workflow"}, bitrate=4000)
anim.save(out_path, writer=writer, dpi=100)
plt.close(fig)
print("wrote", out_path)
