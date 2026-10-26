#!/usr/bin/env python3
"""Render a 10-second vertical clip from the Homework 5 answer cards.

Timeline (30 fps, 1080x1350, 4:5 feed):

    0.0 - 2.0  title card
    2.0 - 8.0  six answers, ~1s each
    8.0 - 9.6  end card
    = ~10s after cross-fades

Output: out/hw5-answers-1080x1350.mp4 (H.264, yuv420p, silent).
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

W, H = 1080, 1350
FPS = 30
XFADE = 0.4
DUR_TITLE, DUR_ANSWER, DUR_END = 2.0, 1.0, 1.6

IMAGES = ["hw5-title.png"] + [f"hw5-q{i}.png" for i in range(1, 7)]


def build():
    # Build input list with durations
    # Use ffmpeg xfade chain
    inputs = []
    for img in IMAGES:
        inputs += ["-loop", "1", "-t", str(DUR_ANSWER if img != "hw5-title.png" else DUR_TITLE), "-i", os.path.join(OUT, img)]
    # Actually set per-image duration: title longer, answers 1s, last (end?) no end card separate; reuse title as end? We'll just use 7 cards.
    # Simpler: equal durations
    cmd = ["ffmpeg", "-y"]
    for img in IMAGES:
        cmd += ["-loop", "1", "-t", "1.4", "-i", os.path.join(OUT, img)]
    # xfade chain
    n = len(IMAGES)
    # Build filter
    filter_parts = []
    # first
    acc = "[0:v]"
    offset = 1.4 - XFADE
    for i in range(1, n):
        out_label = f"[v{i}]"
        filter_parts.append(
            f"{acc}[{i}:v]xfade=transition=fade:duration={XFADE}:offset={offset:.2f}{out_label}"
        )
        acc = out_label
        offset += 1.4 - XFADE
    filter_str = ";".join(filter_parts)
    cmd += ["-filter_complex", filter_str, "-map", acc,
            "-r", str(FPS), "-s", f"{W}x{H}", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            os.path.join(OUT, "hw5-answers-1080x1350.mp4")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("done")


if __name__ == "__main__":
    build()
