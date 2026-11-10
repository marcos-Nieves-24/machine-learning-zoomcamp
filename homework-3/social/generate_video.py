#!/usr/bin/env python3
"""Render a 10-second vertical clip from the Homework 3 answer cards.

Why this exists instead of wiring one of the video harnesses: this machine has
ffmpeg 6.1.1 with libx264, xfade and zoompan, and the card design system
already lives in ``generate_social_cards.py``. A 10 second clip is one title,
six answer cards and one end card, cross-faded with a Ken Burns push. That is
~40 lines of ffmpeg against a Docker image, an Ollama install and a Piper
voice download.

Timeline (30 fps, 1080x1920, 9:16):

    0.0 - 2.2  title card
    2.2 - 8.2  six answers, 1.5s each
    8.2 - 9.8  end card
    = 10.0s total after seven 0.4s cross-fades

Output: out/hw3-answers-1080x1920.mp4 (H.264, yuv420p, silent AAC track).
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_social_cards as cards  # noqa: E402  (design system, reused)

W, H = 1080, 1920
FPS = 30
XFADE = 0.4
DUR_TITLE, DUR_ANSWER, DUR_END = 2.2, 1.5, 1.6
ANSWERS = cards.ANSWERS
NOTES = {
    1: "NA tiene 240 filas y no es la moda",
    2: "r = 0.92, la más alta del dataset",
    3: "MI = 0.03, la más alta contra el target",
    4: "645 de 1000 leads de validación",
    5: "interaction_count cuesta 0.044",
    6: "con C = 0.000001 cae a 0.598",
}
INK, SURFACE, HAIRLINE = cards.INK, cards.SURFACE, cards.HAIRLINE
TEXT, MUTED, FAINT, GOLD = cards.TEXT, cards.MUTED, cards.FAINT, cards.GOLD
REG, MED, SEMI, BOLD, XBOLD = cards.REG, cards.MED, cards.SEMI, cards.BOLD, cards.XBOLD
MONO = cards.MONO
MARGIN = 90


def eyebrow(c, y, s, size=20):
    c.text(MARGIN, y, s, size, SEMI, MUTED, tracking=1)


def progress(c, y_top, active, total=6):
    """Six small segments at the bottom: feed viewers watch muted, so the
    position has to be visible without sound."""
    seg_w, gap = 132, 14
    x = MARGIN
    for i in range(total):
        on = i < active
        c.rect(x, y_top, seg_w, 6, GOLD if on else HAIRLINE, r=3, zorder=6,
               alpha=1.0 if on else 0.9)
        x += seg_w + gap


def scene_title(c):
    cards.draw_background(c)
    eyebrow(c, 210, "ML ZOOMCAMP 2026 · HOMEWORK 3")
    title = "Las 6 respuestas"
    size = c.fit(title, 116, XBOLD, W - 2 * MARGIN, min_size=60)
    c.text(MARGIN, 380, title, size, XBOLD, TEXT)
    c.text(MARGIN, 380 + int(size * 0.62), "Clasificación · lead scoring", 38, MED, MUTED)
    c.text(MARGIN, 380 + int(size * 0.62) + 62, "5,000 leads · 57.7% convirtieron", 32, MED, FAINT)
    c.pill(W - MARGIN, 232, "ANSWER KEY", size=20, fg=INK, bg=GOLD, border=GOLD)
    progress(c, 1700, 0)


def scene_answer(c, i, a):
    cards.draw_background(c)
    eyebrow(c, 200, f"PREGUNTA {a.n} DE 6")
    c.text(MARGIN, 320, f"{a.n:02d}", 92, XBOLD, GOLD)
    c.text(MARGIN, 450, a.question, 40, MED, MUTED)

    size = c.fit(a.value, 118, XBOLD, W - 2 * MARGIN, min_size=46)
    c.text(MARGIN, 620, a.value, size, XBOLD, GOLD)

    # pill sits under the answer, flush with the left margin
    chip_size = 30
    chip_w = c.measure(a.chip, chip_size, SEMI, tracking=0.6)[0] + 52
    c.pill(MARGIN + chip_w, 620 + int(size * 0.95), a.chip, size=chip_size,
               fg=MUTED, bg=SURFACE, border=HAIRLINE, pad_x=26, pad_y=14)

    # one line of context, so a muted viewer still gets the reason
    note_y = 1180
    c.rule(MARGIN, note_y, 120, GOLD, 4, zorder=6)
    for line in c.wrap(NOTES[a.n], 34, MED, W - 2 * MARGIN):
        c.text(MARGIN, note_y + 60, line, 34, MED, MUTED, zorder=6)
        note_y += 52
    progress(c, 1700, i)
    progress(c, 1700, i)


def scene_end(c):
    cards.draw_background(c)
    eyebrow(c, 200, "ML ZOOMCAMP 2026 · HOMEWORK 3")
    c.text(MARGIN, 400, "¿Cuántas sacaste", 92, XBOLD, TEXT)
    c.text(MARGIN, 520, "bien?", 92, XBOLD, TEXT)
    # no emoji: Inter has no emoji glyphs, so the character renders as a box
    c.text(MARGIN, 660, "Contalo en los comentarios", 40, MED, MUTED)
    c.text(MARGIN, 760, "accuracy 0.645 · mejor C: 0.001", 32, MED, FAINT)
    progress(c, 1700, 6)


def render_keyframes(tmp):
    os.makedirs(tmp, exist_ok=True)
    paths = []
    jobs = [("title", scene_title, None)] + \
           [("answer", scene_answer, (i + 1, a)) for i, a in enumerate(ANSWERS)] + \
           [("end", scene_end, None)]
    for name, fn, arg in jobs:
        c = cards.Canvas(W, H)
        cards.register_fonts()
        cards.register_mono()
        (fn(c, *arg) if arg else fn(c))
        p = os.path.join(tmp, f"{len(paths):02d}-{name}.png")
        c.save(p)
        paths.append(p)
        print(f"frame {p}")
    return paths


def assemble(paths, out):
    """One Ken Burns push per keyframe, then cross-fade everything together."""
    durs = [DUR_TITLE] + [DUR_ANSWER] * len(ANSWERS) + [DUR_END]
    total = sum(durs) - XFADE * (len(durs) - 1)
    if abs(total - 10.0) > 1e-6:
        raise SystemExit(f"timeline is {total:.3f}s, expected exactly 10.000s")

    filters, inputs = [], []
    for i, (p, d) in enumerate(zip(paths, durs)):
        # -framerate matters: the image loop defaults to 25 fps and the clip
        # would come out short. zoompan d=1 keeps one output frame per input
        # frame, so `on` maps 1:1 to the timeline.
        inputs += ["-framerate", str(FPS), "-loop", "1", "-t", f"{d:.3f}", "-i", p]
        frames = int(round(d * FPS))
        zoom = f"1+0.055*on/{frames - 1}"
        filters.append(
            f"[{i}:v]scale={W}:{H},setsar=1,"
            f"zoompan=z='{zoom}':d=1:x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}[v{i}]"
        )

    offset = 0.0
    prev = "v0"
    for i in range(1, len(paths)):
        offset += durs[i - 1] - XFADE
        label = f"x{i}" if i < len(paths) - 1 else "vout"
        # fadeblack, not fade: two text-heavy scenes cross-faded into each
        # other read as overlap noise, a dip to black reads as an edit.
        filters.append(
            f"[{prev}][v{i}]xfade=transition=fadeblack:duration={XFADE}:"
            f"offset={offset:.3f}[{label}]"
        )
        prev = label

    cmd = ["ffmpeg", "-y", *inputs,
           "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
           "-filter_complex", ";".join(filters),
           "-map", "[vout]", "-map", f"{len(paths)}:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "128k", "-shortest",
           "-movflags", "+faststart", out]
    print("\n" + " ".join(cmd[:12]) + " …\n")
    subprocess.run(cmd, check=True)
    return out


def probe(out):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_name,width,height,r_frame_rate,nb_frames",
         "-of", "json", out], capture_output=True, text=True, check=True)
    return r.stdout


def main():
    cards.register_fonts()
    cards.register_mono()
    tmp = os.path.join(HERE, "tmp-video")
    out = os.path.join(HERE, "out", "hw3-answers-1080x1920.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    paths = render_keyframes(tmp)
    assemble(paths, out)
    print("\n--- ffprobe ---")
    print(probe(out))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
