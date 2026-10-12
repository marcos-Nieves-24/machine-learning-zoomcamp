#!/usr/bin/env python3
"""Generate publishable social cards for ML Zoomcamp 2026, Homework 3.

Concept: an "answer key" card. Deep ink background, gold marks for the answers
(the highlighter metaphor), quiet neutral grays for structure. Typography is
Inter (bundled in ./fonts, SIL OFL 1.1) so the output looks the same anywhere.

Every number rendered here is taken from the executed ``homework-3.ipynb``.
Nothing is hand-typed or invented.

Usage:
    python3 generate_social_cards.py            # renders out/<platform>-*.png
    python3 generate_social_cards.py --preview  # also writes a contact sheet

Canvas geometry: 1 design unit == 1 output pixel. Matplotlib font sizes are in
points, so a design size of ``px`` becomes ``px * 72 / DPI`` points.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

DPI = 200  # physical pixels per output pixel at savefig time
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

INK = "#0B0E14"  # page background
SURFACE = "#12161F"  # raised panels
HAIRLINE = "#222A37"  # 1px rules and borders
TEXT = "#F3F5F9"  # primary text
MUTED = "#8D97A8"  # secondary text
FAINT = "#5A6472"  # tertiary text
GOLD = "#F5C44C"  # accent: the answers (the "highlighter")
GOLD_DIM = "#8A6E22"
COOL = "#5E8BFF"  # ambient glow only

REG, MED, SEMI, BOLD, XBOLD, BLACK = 400, 500, 600, 700, 800, 900
THIN_SPACE = "\u2009"  # U+2009, used to fake letter-spacing on Inter


def pt(px: float) -> float:
    """Design pixels -> matplotlib points at the working DPI."""
    return px * 72.0 / DPI


# --------------------------------------------------------------------------- #
# The six answers (data of record)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Answer:
    n: int
    question: str
    value: str
    chip: str


ANSWERS = [
    Answer(1, "Moda de industry", "technology", "1,173 filas"),
    Answer(2, "Mayor correlación", "interaction_count × lead_score", "r = 0.92"),
    Answer(3, "Mayor información mutua", "lead_source", "MI = 0.03"),
    Answer(4, "Accuracy en validación", "0.65", "raw 0.645"),
    Answer(5, "Feature menos útil", "number_of_courses_viewed", "Δ = 0.002"),
    Answer(6, "Mejor valor de C", "0.001", "acc 0.645"),
]

EYEBROW = "ML ZOOMCAMP 2026 · HOMEWORK 3"
TITLE = "Clasificación: las 6 respuestas"
SUBTITLE = "Lead scoring · 5,000 leads · 57.7% convirtieron · split 60/20/20"
FOOTER_LEFT = "03-classification · LogisticRegression (liblinear, C=1.0)"
FOOTER_RIGHT = "DataTalksClub/machine-learning-zoomcamp"


def register_fonts() -> None:
    """Register the bundled Inter faces with matplotlib's font manager."""
    faces = [
        "Inter-Regular.ttf",
        "Inter-Medium.ttf",
        "Inter-SemiBold.ttf",
        "Inter-Bold.ttf",
        "Inter-ExtraBold.ttf",
        "Inter-Black.ttf",
        "InterDisplay-Bold.ttf",
    ]
    for name in faces:
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing font: {path}")
        fm.fontManager.addfont(path)
    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["font.weight"] = REG
    plt.rcParams["axes.unicode_minus"] = False


def register_mono() -> None:
    """Register the bundled Fira Code faces (SIL OFL 1.1) for code panels."""
    global MONO_READY
    faces = [
        "FiraCode-Regular.ttf",
        "FiraCode-Medium.ttf",
        "FiraCode-SemiBold.ttf",
        "FiraCode-Bold.ttf",
    ]
    for name in faces:
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing font: {path}")
        fm.fontManager.addfont(path)
    MONO_READY = True


# --------------------------------------------------------------------------- #
# Canvas primitives
# --------------------------------------------------------------------------- #


class Canvas:
    """A pixel-exact drawing surface with the handful of primitives a card needs."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor(INK)
        self.renderer = self.fig.canvas.get_renderer()

    # -- coordinates -------------------------------------------------------- #
    def y(self, from_top: float) -> float:
        """Convert a top-down coordinate to matplotlib's bottom-up space."""
        return self.h - from_top

    # -- text --------------------------------------------------------------- #
    def text(self, x, y_top, s, size, weight=REG, color=TEXT, ha="left",
             tracking=0.0, alpha=1.0, zorder=6, family="Inter"):
        if tracking:
            s = THIN_SPACE.join(s) if tracking == 1 else " ".join(s)
        return self.ax.text(
            x, self.y(y_top), s,
            fontsize=pt(size), fontweight=weight, color=color,
            ha=ha, va="baseline", alpha=alpha, zorder=zorder, family=family,
        )

    def measure(self, s, size, weight=REG, tracking=0.0, family="Inter"):
        """Width/height of a string in design pixels."""
        if tracking:
            s = THIN_SPACE.join(s) if tracking == 1 else " ".join(s)
        t = self.ax.text(0, 0, s, fontsize=pt(size), fontweight=weight, family=family)
        bb = t.get_window_extent(renderer=self.renderer)
        inv = self.ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        t.remove()
        return x1 - x0, y1 - y0

    def fit(self, s, size, weight, max_w, min_size=18):
        """Shrink a font size until the string fits, never below ``min_size``."""
        while size > min_size:
            w, _ = self.measure(s, size, weight)
            if w <= max_w:
                break
            size -= 1
        return size

    def wrap(self, s, size, weight, max_w, family="Inter"):
        """Greedy word wrap measured against the real font metrics."""
        words, lines, cur = s.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            w, _ = self.measure(trial, size, weight, family=family)
            if w <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    # -- shapes ------------------------------------------------------------- #
    def rect(self, x, y_top, w, h, fc, ec="none", lw=0, r=0, zorder=2, alpha=1.0):
        if r <= 0:
            p = Rectangle((x, self.y(y_top) - h), w, h, facecolor=fc,
                          edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha)
        else:
            p = FancyBboxPatch(
                (x, self.y(y_top) - h), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha,
                mutation_aspect=1,
            )
        self.ax.add_patch(p)
        return p

    def glow(self, cx, cy_top, radius, color, strength=0.16):
        """Soft radial light, blended additively onto the background."""
        n = 220
        yy, xx = np.mgrid[0:n, 0:n]
        xx = xx / (n - 1) * self.w
        yy = self.h - (yy / (n - 1) * self.h)
        d = np.sqrt((xx - cx) ** 2 + (yy - self.y(cy_top)) ** 2) / radius
        fall = np.clip(1.0 - d, 0.0, 1.0) ** 2.2
        rgba = np.zeros((n, n, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(color)
        rgba[..., 3] = fall * strength
        self.ax.imshow(rgba, extent=[0, self.w, 0, self.h], origin="upper",
                       zorder=1, interpolation="bilinear", aspect="auto")

    def rule(self, x, y_top, w, color=HAIRLINE, thickness=1, zorder=3, alpha=1.0):
        self.rect(x, y_top, w, thickness, color, zorder=zorder, alpha=alpha)

    def gradient_rule(self, x, y_top, width, thickness=3, zorder=3):
        """Hairline that fades in and out, like a printed divider."""
        n = 256
        alpha = np.sin(np.linspace(0, np.pi, n)) ** 1.4  # 0 -> 1 -> 0
        rgba = np.zeros((2, n, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(GOLD)
        rgba[..., 3] = alpha
        self.ax.imshow(rgba, extent=[x, x + width, self.y(y_top) - thickness, self.y(y_top)],
                       aspect="auto", zorder=zorder, origin="lower", interpolation="bilinear")

    # -- composites --------------------------------------------------------- #
    def check_bounds(self, margin: int, name: str) -> list[str]:
        """Fail loudly if any text element escapes the safe area.

        Guards against the two classic card bugs: text clipped by the canvas
        edge, and text crossing the side margins (i.e. colliding with a chip).
        """
        tol = 1.5
        problems = []
        for t in self.ax.texts:
            if not t.get_text():
                continue
            bb = t.get_window_extent(renderer=self.renderer)
            inv = self.ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if x0 < margin - tol or x1 > self.w - margin + tol:
                problems.append(
                    f"{name}: '{t.get_text()[:38]}' x=[{x0:.0f},{x1:.0f}] "
                    f"outside [{margin},{self.w - margin}]")
            if y0 < -tol or y1 > self.h + tol:
                problems.append(f"{name}: '{t.get_text()[:38]}' leaves the canvas")
        for p in problems:
            print("  OVERFLOW", p)
        return problems

    def pill(self, x_right, y_center_top, s, size=16, fg=GOLD, bg=SURFACE,
             border=HAIRLINE, pad_x=16, pad_y=9, weight=SEMI, tracking=0.6):
        """Right-aligned pill; returns (left_x, top_y, width, height)."""
        tw, _ = self.measure(s, size, weight, tracking=tracking)
        w = tw + 2 * pad_x
        h = size + 2 * pad_y
        x = x_right - w
        top = y_center_top - h / 2
        self.rect(x, top, w, h, bg, ec=border, lw=1, r=h / 2, zorder=5)
        self.text(x + w / 2, top + h / 2 + size * 0.35, s, size, weight, fg,
                  ha="center", tracking=tracking, zorder=6)
        return x, top, w, h

    def save(self, path):
        self.fig.savefig(path, dpi=DPI, facecolor=INK, edgecolor="none")
        plt.close(self.fig)
        # Social platforms expect opaque RGB: flatten the alpha channel away
        # instead of shipping a translucent PNG some uploaders render badly.
        from PIL import Image

        im = Image.open(path)
        if im.mode == "RGBA":
            flat = Image.new("RGB", im.size, INK)
            flat.paste(im, mask=im.split()[3])
            flat.save(path, optimize=True)
        return path


# --------------------------------------------------------------------------- #
# Card sections
# --------------------------------------------------------------------------- #


def draw_background(c: Canvas):
    """Ink page + two soft light sources so the surface is never flat black."""
    c.glow(c.w * 0.12, -c.h * 0.05, c.w * 0.95, GOLD, strength=0.13)
    c.glow(c.w * 1.05, c.h * 1.06, c.w * 0.85, COOL, strength=0.10)


def draw_header(c: Canvas, margin: int, top: int, kicker_size: int,
                title_size: int, sub_size: int, pill_label: str,
                title: str | None = None, subtitle: str | None = None):
    """Eyebrow + pill on one line, then title, then subtitle. Returns the y of
    the lowest element (top-down)."""
    title = TITLE if title is None else title
    subtitle = SUBTITLE if subtitle is None else subtitle

    eyebrow_y = top + kicker_size
    c.text(margin, eyebrow_y, EYEBROW, kicker_size, SEMI, MUTED, tracking=1)

    if pill_label:
        # vertically centred on the eyebrow line, flush right
        c.pill(c.w - margin, eyebrow_y - kicker_size * 0.35, pill_label,
               size=max(13, kicker_size - 1), fg=INK, bg=GOLD, border=GOLD)

    avail = c.w - 2 * margin
    tsize = c.fit(title, title_size, XBOLD, avail, min_size=int(title_size * 0.7))
    # 1.12 * size keeps the title's cap height clear of the pill's lower edge
    title_y = eyebrow_y + int(tsize * 1.12)
    c.text(margin, title_y, title, tsize, XBOLD, TEXT)

    sub_y = title_y + int(tsize * 0.58)
    c.text(margin, sub_y, subtitle, sub_size, MED, MUTED)

    return sub_y + sub_size


def draw_rule(c: Canvas, margin: int, y_top: int, width: int, thickness=3):
    c.gradient_rule(margin, y_top, width, thickness)


def draw_footer(c: Canvas, margin: int, base: int, size: int):
    c.rule(margin, base - size * 2.1, c.w - 2 * margin, HAIRLINE, 1)
    c.text(margin, base, FOOTER_LEFT, size, MED, FAINT)
    c.text(c.w - margin, base, FOOTER_RIGHT, size, MED, FAINT, ha="right")


PROBLEMS: list[str] = []


def finish(c: Canvas, path: str, margin: int):
    """Verify the layout, then write the card out."""
    PROBLEMS.extend(c.check_bounds(margin, os.path.basename(path)))
    return c.save(path)


def draw_rows(c: Canvas, margin: int, top: int, bottom: int,
              num_size: int, label_size: int, value_size: int, chip_size: int,
              numerals: bool = True, chips: bool = True,
              chip_mode: str = "right", divisors: bool = True):
    """Six answer rows, laid out between ``top`` and ``bottom`` (top-down px)."""
    n = len(ANSWERS)
    area = bottom - top
    row_h = area / n
    num_col = int(num_size * 1.9) if numerals else 0
    chip_w = 0
    if chips and chip_mode == "right":
        chip_w = max(c.measure(a.chip, chip_size, SEMI, tracking=0.6)[0] for a in ANSWERS) + 34
    content_x = margin + num_col
    content_w = c.w - margin - content_x - (chip_w + 24 if chip_w else 0)

    for i, a in enumerate(ANSWERS):
        row_top = top + i * row_h
        if divisors and i > 0:
            c.rule(content_x, row_top, c.w - margin - content_x, HAIRLINE, 1, alpha=0.85)

        if numerals:
            c.text(margin, row_top + row_h / 2 + num_size * 0.35,
                   f"{a.n:02d}", num_size, XBOLD, GOLD)

        label_y = row_top + row_h / 2 - label_size * 0.95
        c.text(content_x, label_y, a.question, label_size, MED, MUTED)

        vsize = c.fit(a.value, value_size, BOLD, content_w, min_size=int(value_size * 0.6))
        value_y = label_y + int(label_size * 1.85) + int(vsize * 0.15)
        c.text(content_x, value_y, a.value, vsize, BOLD, GOLD)

        if chips and chip_mode == "right":
            c.pill(c.w - margin, label_y + label_size * 0.9, a.chip,
                   size=chip_size, fg=MUTED, bg=SURFACE, border=HAIRLINE)
        elif chips:
            c.text(content_x, value_y + vsize * 1.45, a.chip, chip_size, SEMI, FAINT)


# --------------------------------------------------------------------------- #
# Platform layouts
# --------------------------------------------------------------------------- #


def card_linkedin(path, w=1200, h=1200):
    c = Canvas(w, h)
    draw_background(c)
    m = 92
    bottom = draw_header(c, m, 92, kicker_size=17, title_size=58, sub_size=20,
                         pill_label="ANSWER KEY")
    draw_rule(c, m, bottom + 34, w - 2 * m)
    draw_footer(c, m, h - 56, size=15)
    draw_rows(c, m, bottom + 34 + 44, h - 56 - 46,
              num_size=26, label_size=19, value_size=29, chip_size=15)
    return finish(c, path, m)


def card_instagram(path, w=1080, h=1350):
    """4:5 portrait, the format that takes the most feed real estate."""
    c = Canvas(w, h)
    draw_background(c)
    m = 84
    bottom = draw_header(c, m, 96, kicker_size=17, title_size=62, sub_size=21,
                         pill_label="ANSWER KEY")
    draw_rule(c, m, bottom + 40, w - 2 * m)
    draw_footer(c, m, h - 64, size=16)
    draw_rows(c, m, bottom + 40 + 52, h - 64 - 52,
              num_size=28, label_size=20, value_size=32, chip_size=16)
    return finish(c, path, m)


def card_instagram_story(path, w=1080, h=1920):
    """9:16 story: same content, airier, with a hook line at the bottom."""
    c = Canvas(w, h)
    draw_background(c)
    m = 80
    bottom = draw_header(c, m, 150, kicker_size=18, title_size=64, sub_size=22,
                         pill_label="ANSWER KEY")
    draw_rule(c, m, bottom + 48, w - 2 * m)
    draw_footer(c, m, h - 190, size=16)
    draw_rows(c, m, bottom + 48 + 60, h - 190 - 56,
              num_size=30, label_size=22, value_size=34, chip_size=17)
    c.text(m, h - 118, "¿Cuántas sacaste bien? Contame en los comentarios.",
           24, SEMI, TEXT)
    c.text(m, h - 78, "Guardá este post para el próximo intento.",
           19, MED, FAINT)
    return finish(c, path, m)


def card_twitter(path, w=1600, h=900):
    """16:9 landscape, two columns of three so type stays large."""
    c = Canvas(w, h)
    draw_background(c)
    m = 88
    bottom = draw_header(c, m, 78, kicker_size=16, title_size=48, sub_size=18,
                         pill_label="ANSWER KEY")
    draw_rule(c, m, bottom + 30, w - 2 * m)
    draw_footer(c, m, h - 48, size=15)

    col_gap = 56
    col_w = (w - 2 * m - col_gap) / 2
    rows_top = bottom + 30 + 44
    rows_bottom = h - 48 - 40
    row_h = (rows_bottom - rows_top) / 3

    for col in range(2):
        x0 = m + col * (col_w + col_gap)
        for r in range(3):
            a = ANSWERS[col * 3 + r]
            row_top = rows_top + r * row_h
            if r > 0:
                c.rule(x0, row_top, col_w, HAIRLINE, 1, alpha=0.85)
            c.text(x0, row_top + row_h / 2 + 24 * 0.35, f"{a.n:02d}", 24, XBOLD, GOLD)
            content_x = x0 + 46
            content_w = col_w - 46
            label_y = row_top + row_h / 2 - 15 * 0.95
            c.text(content_x, label_y, a.question, 15, MED, MUTED)
            vsize = c.fit(a.value, 27, BOLD, content_w, min_size=17)
            value_y = label_y + int(15 * 1.85) + int(vsize * 0.15)
            c.text(content_x, value_y, a.value, vsize, BOLD, GOLD)
            c.text(content_x, value_y + vsize * 1.5, a.chip, 13, SEMI, FAINT)
    return finish(c, path, m)


CARDS = [
    ("linkedin-post-1200x1200.png", card_linkedin),
    ("instagram-post-1080x1350.png", card_instagram),
    ("instagram-story-1080x1920.png", card_instagram_story),
    ("x-card-1600x900.png", card_twitter),
]


# --------------------------------------------------------------------------- #
# Code snippets (LinkedIn series)
# --------------------------------------------------------------------------- #

try:  # optional dependency: without Pygments the code renders in one colour
    from pygments.lexers import PythonLexer
    from pygments.token import (
        Comment, Keyword, Name, Number, Operator, String, Text,
    )

    HAVE_PYGMENTS = True
except ImportError:  # pragma: no cover
    HAVE_PYGMENTS = False

MONO = "Fira Code"
CODE_TEXT = "#DDE4EE"
SYNTAX = {
    "keyword": "#C4A2FF",   # keywords: import, for, in, def, return
    "string": "#93DDA0",    # string literals
    "number": "#FFB770",    # numeric literals
    "comment": "#68738A",   # comments
    "func": "#7FB8FF",      # function and builtin names
    "operator": "#9AA6B8",  # operators and punctuation
    "text": CODE_TEXT,      # everything else
}


def _token_color(tok) -> str:
    if not HAVE_PYGMENTS:
        return CODE_TEXT
    if tok in Comment:
        return SYNTAX["comment"]
    if tok in Keyword:
        return SYNTAX["keyword"]
    if tok in String:
        return SYNTAX["string"]
    if tok in Number:
        return SYNTAX["number"]
    if tok in Name.Function or tok in Name.Builtin:
        return SYNTAX["func"]
    if tok in Operator:
        return SYNTAX["operator"]
    return SYNTAX["text"]


def highlight(code: str):
    """Split source into per-line runs of (colour, text).

    Monospace means every glyph advances the same width, so a run can be
    positioned by counting characters instead of measuring each piece.
    """
    if not HAVE_PYGMENTS:
        return [[(CODE_TEXT, line)] for line in code.split("\n")]

    lines: list[list[tuple[str, str]]] = [[]]
    for tok, text in PythonLexer().get_tokens(code):
        color = _token_color(tok)
        for i, part in enumerate(text.split("\n")):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((color, part))
    while lines and not lines[-1]:  # Pygments appends a trailing newline
        lines.pop()
    return lines


def draw_code_card(path, *, filename, title, subtitle, code, result, note=None,
                   w=1200, h=1200, m=78):
    c = Canvas(w, h)
    draw_background(c)
    bottom = draw_header(c, m, 88, kicker_size=17, title_size=54, sub_size=19,
                         pill_label="PYTHON", title=title, subtitle=subtitle)

    lines = highlight(code.strip("\n"))
    n_lines = len(lines)
    longest = max((sum(len(t) for _, t in line) for line in lines), default=0)

    footer_ledge = 66  # space reserved for the footer rule and text
    result_h, gap = 58, 34
    note_size, note_lh = 17, 26

    # Pick the largest code size whose longest line still fits the panel.
    pad, gutter, head_h = 26, 40, 48
    inner_w = w - 2 * m - 2 * pad - gutter
    for size in (34, 32, 30, 28, 26, 24, 22, 20):
        advance, _ = c.measure("0" * 24, size, REG, family=MONO)
        advance /= 24
        if longest * advance <= inner_w:
            break
    line_h = size * 1.5

    # Optional note under the result strip, wrapped to the content width.
    note_lines = c.wrap(note, note_size, MED, w - 2 * m) if note else []
    note_h = len(note_lines) * note_lh + 18 if note_lines else 0

    region_top = bottom + 40
    region_bottom = h - footer_ledge
    # Room the panel may take once the result strip and note are placed.
    room = region_bottom - region_top - gap - result_h - (gap + note_h if note_h else 0)

    def panel_height(lh):
        return head_h + pad + (n_lines * lh - (lh - size)) + pad

    # Short snippets get airier leading instead of a dead band under the panel.
    while panel_height(line_h) < room and line_h < size * 2.4:
        line_h += 2
    panel_h = panel_height(line_h)
    if panel_h > room:  # last resort: shrink the code until it fits
        while panel_h > room and size > 16:
            size -= 2
            line_h = size * 1.5
            panel_h = panel_height(line_h)

    block_h = panel_h + gap + result_h + (gap + note_h if note_h else 0)
    slack = max(0, (region_bottom - region_top - block_h) // 3)
    panel_top = region_top + slack

    c.rect(m, panel_top, w - 2 * m, panel_h, SURFACE, ec=HAIRLINE, lw=1, r=18, zorder=4)

    # panel chrome: filename on the left, language on the right
    c.text(m + pad, panel_top + 30, filename, 15, MED, MUTED, zorder=6, family=MONO)
    c.pill(m + (w - 2 * m) - pad, panel_top + 24, "python", size=13, fg=GOLD,
           bg=SURFACE, border=HAIRLINE, pad_x=12, pad_y=6)
    c.rule(m + pad, panel_top + head_h, w - 2 * m - 2 * pad, HAIRLINE, 1, zorder=5)

    # code
    code_x = m + pad + gutter
    y = panel_top + head_h + pad
    for i, line in enumerate(lines, start=1):
        baseline = y + size * 0.8
        c.text(m + pad, baseline, f"{i:2d}", size * 0.62, MED, "#4B5565",
               zorder=6, family=MONO)
        x = code_x
        for color, run in line:
            c.text(x, baseline, run, size, REG, color, zorder=6, family=MONO)
            x += advance * len(run)
        y += line_h

    # result strip: gold wash + gold text, same accent language as the answer cards
    strip_top = panel_top + panel_h + gap
    strip_h = result_h
    c.rect(m, strip_top, w - 2 * m, strip_h, GOLD, r=14, alpha=0.12, zorder=3)
    c.text(m + 24, strip_top + strip_h / 2 + 19 * 0.36, result, 19, SEMI, GOLD, zorder=6)

    if note_lines:
        ny = strip_top + strip_h + gap
        for ln in note_lines:
            c.text(m, ny, ln, note_size, MED, MUTED, zorder=6)
            ny += note_lh

    draw_footer(c, m, h - 34, size=14)
    return finish(c, path, m)


CODE_CARDS = [
    (
        "linkedin-code-01-preparar-los-datos.png",
        dict(
            filename="homework-3.ipynb · data prep",
            title="Preparar los datos",
            subtitle="Faltantes y split 60/20/20 con semillas fijas",
            code='''\
# 1. Faltantes: categoricas -> "NA", numericas -> 0.0
df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

# 2. Split 60/20/20, reproducible
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)''',
            result="0 faltantes  ·  3000 / 1000 / 1000 filas",
            note=("Faltantes en 4 categóricas (lead_source, industry, "
                  "employment_status, location) y en 2 numéricas "
                  "(annual_income, lead_score)."),
        ),
    ),
    (
        "linkedin-code-02-el-modelo.png",
        dict(
            filename="homework-3.ipynb · Q4",
            title="El modelo, en 3 pasos",
            subtitle="One-hot + LogisticRegression + accuracy",
            code='''\
# 1. One-hot de las categoricas
dv = DictVectorizer(sparse=False)
X_tr = dv.fit_transform(train[categorical + numerical])
X_va = dv.transform(val[categorical + numerical])

# 2. El modelo
model = LogisticRegression(
    solver="liblinear", C=1.0, max_iter=1000, random_state=42
)
model.fit(X_tr, y_train)

# 3. Accuracy en validacion
round(accuracy_score(y_val, model.predict(X_va)), 3)''',
            result="val accuracy: 0.645  ->  0.65",
            note=("DictVectorizer ignora las categorías que no aparecen en "
                  "train. 0.645 = 645 de los 1000 leads de validación."),
        ),
    ),
    (
        "linkedin-code-03-feature-menos-util.png",
        dict(
            filename="homework-3.ipynb · Q5",
            title="Que feature sacar sin perder nada",
            subtitle="Feature elimination sobre el accuracy de validacion",
            code='''\
def score(cols):
    d = DictVectorizer(sparse=False)
    Xt = d.fit_transform(train[cols].to_dict("records"))
    Xv = d.transform(val[cols].to_dict("records"))
    m = LogisticRegression(solver="liblinear", C=1.0)
    m.fit(Xt, y_train)
    return accuracy_score(y_val, m.predict(Xv))

features = ["lead_source", "number_of_courses_viewed",
            "interaction_count"]
for f in features:
    cols = [c for c in all_features if c != f]
    print(f"{f:26s}{base - score(cols):+.4f}")''',
            result="menor diferencia: number_of_courses_viewed  (+0.002)",
            note=("Para comparar: sacar interaction_count cuesta 0.044 y sacar "
                  "lead_source, 0.003. La diferencia más chica es la feature "
                  "menos informativa."),
        ),
    ),
    (
        "linkedin-code-04-regularizacion.png",
        dict(
            filename="homework-3.ipynb · Q6",
            title="Regularizar: el barrido de C",
            subtitle="Que tan fuerte conviene frenar los coeficientes",
            code='''\
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    m = LogisticRegression(solver="liblinear", C=C,
                           max_iter=1000, random_state=42)
    m.fit(X_tr, y_train)
    acc = accuracy_score(y_val, m.predict(X_va))
    print(f"{C:<10g}{acc:.3f}")''',
            result="mejor C: 0.001  ·  accuracy 0.645",
            note=("C chico = regularización fuerte. Con C=0.000001 el modelo "
                  "frena tanto que el accuracy cae a 0.598."),
        ),
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default=None,
                    help="render a subset, e.g. --only code or --only answers")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    register_fonts()
    register_mono()

    jobs = []
    if args.only in (None, "answers"):
        jobs += CARDS
    if args.only in (None, "code"):
        jobs += CODE_CARDS

    paths = []
    for name, fn in jobs:
        path = os.path.join(args.outdir, name)
        if isinstance(fn, dict):
            draw_code_card(path, **fn)
        else:
            fn(path)
        paths.append(path)
        print(f"wrote {path}")

    if not HAVE_PYGMENTS:
        print("note: pygments not installed, code rendered without syntax colour")

    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} overflow problem(s) detected")
        raise SystemExit(1)
    print(f"\n{len(paths)} cards in {args.outdir}, no overflow")


if __name__ == "__main__":
    main()
