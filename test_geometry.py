#!/usr/bin/env python3
"""Tests for the crop rules. No torch needed — that is why geometry.py is separate.

    python3 test_geometry.py
"""

import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "h3geom", os.path.join(_here, "h3_audio", "geometry.py"))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

_spec = importlib.util.spec_from_file_location(
    "h3timing", os.path.join(_here, "h3_audio", "timing.py"))
t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)

fails = []


def eq(got, want, what):
    if got != want:
        fails.append(f"{what}: got {got}, want {want}")


def approx_aspect(cw, ch, tw, th, what):
    # the crop is integral, so it can only hit the target ratio to within a pixel
    if abs(cw / ch - tw / th) > 1.5 / min(cw, ch):
        fails.append(f"{what}: {cw}x{ch} is {cw / ch:.4f}:1, want {tw / th:.4f}:1")


# --- crop_to_multiple -------------------------------------------------------
eq(g.crop_to_multiple(640, 1120), (0, 0, 640, 1120), "already aligned is a no-op")
eq(g.crop_to_multiple(1024, 574), (0, 15, 1024, 544), "574 -> 544, centred")
eq(g.crop_to_multiple(1920, 1080), (0, 12, 1920, 1056), "1080 -> 1056")
eq(g.crop_to_multiple(1000, 1000), (4, 4, 992, 992), "both axes trimmed")
for w, h in ((1920, 1080), (1000, 1000), (33, 65), (640, 361)):
    x0, y0, cw, ch = g.crop_to_multiple(w, h)
    if cw % 32 or ch % 32:
        fails.append(f"crop_to_multiple({w},{h}) -> {cw}x{ch}, not /32")
    if x0 + cw > w or y0 + ch > h or x0 < 0 or y0 < 0:
        fails.append(f"crop_to_multiple({w},{h}) -> rect escapes the source")
try:
    g.crop_to_multiple(20, 500)
    fails.append("crop_to_multiple should reject a source under 32px")
except ValueError:
    pass

# --- cover_crop -------------------------------------------------------------
eq(g.cover_crop(640, 1120, 640, 1120), (0, 0, 640, 1120), "identical is a no-op")
eq(g.cover_crop(1280, 720, 640, 360), (0, 0, 1280, 720), "same aspect, no crop")
eq(g.cover_crop(1920, 1080, 640, 1120), (651, 0, 617, 1080), "16:9 into portrait")
eq(g.cover_crop(640, 1120, 1024, 576), (0, 380, 640, 360), "portrait into 16:9")

for sw, sh, tw, th in ((1920, 1080, 640, 1120), (640, 1120, 1024, 576),
                       (1024, 574, 768, 768), (3840, 2160, 1344, 768),
                       (720, 1280, 640, 1120)):
    x0, y0, cw, ch = g.cover_crop(sw, sh, tw, th)
    what = f"cover_crop({sw},{sh} -> {tw},{th})"
    approx_aspect(cw, ch, tw, th, what)
    if x0 + cw > sw or y0 + ch > sh or x0 < 0 or y0 < 0:
        fails.append(f"{what}: rect escapes the source")
    if cw != sw and ch != sh:
        fails.append(f"{what}: cropped BOTH axes; cover should trim only one")
    # centred: the discarded margins differ by at most a pixel
    if abs((sw - cw - x0) - x0) > 1 or abs((sh - ch - y0) - y0) > 1:
        fails.append(f"{what}: not centred")

# --- timing -----------------------------------------------------------------
eq(t.av_aligned_runs_through(400), [39, 90, 141, 192, 243, 294, 345, 396],
   "AV-aligned runs")
for n in t.av_aligned_runs_through(1500):
    if (n - 5) % 17:
        fails.append(f"{n} is AV-aligned but not a legal video run")
    if t.audio_t(n) != n * 40 / 24:
        fails.append(f"{n}: audio_t rounds, so it is not really aligned")
eq(t.is_av_aligned(362), False, "362 is off the audio grid")
eq(t.snap_av_aligned(362), 345, "362 snaps down to 345")
eq(t.snap_av_aligned(362, "up"), 396, "362 snaps up to 396")
eq(t.snap_av_aligned(345), 345, "an aligned run is left alone")
eq(t.snap_av_aligned(5, "down"), 39, "never below the smallest aligned run")
eq(t.align_frames(14.375), 345, "14.375 s is exactly 345 frames")
eq(t.align_frames(29.25), 702, "29.25 s is exactly 702 frames")
eq(round(t.av_error_steps(362), 3), 0.333, "362 is a third of a step out")

if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all geometry + timing tests pass")
