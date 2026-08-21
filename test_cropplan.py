#!/usr/bin/env python3
"""Tests for the subject-crop planner. No torch needed.

    python3 test_cropplan.py
"""

import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "cropplan", os.path.join(_here, "cropplan.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

W, H = 1280, 720
fails = []


def bad(msg):
    fails.append(msg)


def contains(box, bb):
    return (box["x"] <= bb[0] and box["y"] <= bb[1]
            and bb[2] < box["x"] + box["width"]
            and bb[3] < box["y"] + box["height"])


def check(boxes, bboxes, what, div=32):
    for i, (b, bb) in enumerate(zip(boxes, bboxes)):
        if bb is None:
            continue
        if not contains(b, bb):
            bad(f"{what}: frame {i} subject {bb} escapes box {b}")
            return
        if b["width"] % div or b["height"] % div:
            bad(f"{what}: frame {i} box {b['width']}x{b['height']} not /{div}")
            return
        if b["x"] < 0 or b["y"] < 0 or b["x"] + b["width"] > W or b["y"] + b["height"] > H:
            bad(f"{what}: frame {i} box {b} leaves the image")
            return


# --- a still subject -------------------------------------------------------
still = [(600, 300, 700, 600)] * 60
for mode in ("static", "tracked"):
    boxes, info = cp.plan(still, W, H, mode=mode, crop_scale=1.5)
    check(boxes, still, f"still/{mode}")
    if info["moves"] != 0:
        bad(f"still/{mode}: crop moved {info['moves']} times on a static subject")
    if not info["covered"]:
        bad(f"still/{mode}: reported not covered")
    if boxes[0]["width"] * boxes[0]["height"] >= W * H:
        bad(f"still/{mode}: box is not smaller than the frame")

# --- jitter must NOT move a tracked crop -----------------------------------
# +/-3 px of segmentation noise, no real motion
jit = [(600 + (i % 3) - 1, 300, 700 + (i % 3) - 1, 600) for i in range(60)]
boxes, info = cp.plan(jit, W, H, mode="tracked", crop_scale=1.5, smooth_window=16)
check(boxes, jit, "jitter/tracked")
if info["moves"] > 0:
    bad(f"jitter/tracked: {info['moves']} moves from noise alone — hysteresis failed")

# --- sustained motion MUST move it -----------------------------------------
walk = [(100 + i * 12, 300, 200 + i * 12, 600) for i in range(60)]
boxes, info = cp.plan(walk, W, H, mode="tracked", crop_scale=1.5)
check(boxes, walk, "walk/tracked")
if info["moves"] == 0:
    bad("walk/tracked: crop never moved while the subject crossed the frame")
if boxes[0]["width"] * boxes[0]["height"] >= W * H:
    bad("walk/tracked: box is not smaller than the frame")
# static over the same walk must cover the whole path, so it should be bigger
sboxes, sinfo = cp.plan(walk, W, H, mode="static", crop_scale=1.5)
check(sboxes, walk, "walk/static")
if sboxes[0]["width"] <= boxes[0]["width"]:
    bad("walk/static: static box should be wider than tracked on a crossing subject")

# --- empty frames ----------------------------------------------------------
gappy = [(600, 300, 700, 600)] * 20 + [None] * 5 + [(600, 300, 700, 600)] * 20
for mode in ("static", "tracked"):
    boxes, info = cp.plan(gappy, W, H, mode=mode, crop_scale=1.5)
    check(boxes, gappy, f"gaps/{mode}")
    if len(boxes) != len(gappy):
        bad(f"gaps/{mode}: {len(boxes)} boxes for {len(gappy)} frames")
try:
    cp.plan([None] * 10, W, H)
    bad("an all-empty mask should raise, not plan a box")
except ValueError:
    pass

# --- disabled --------------------------------------------------------------
boxes, info = cp.plan(still, W, H, crop_scale=0.0)
if info["mode"] != "disabled":
    bad("crop_scale 0 should report disabled")
if boxes[0] != {"x": 0, "y": 0, "width": 1280, "height": 704}:
    bad(f"crop_scale 0 should pass the full frame on the grid, got {boxes[0]}")
if len(boxes) != len(still):
    bad("disabled should still emit one box per frame")

# --- aspect lock -----------------------------------------------------------
for ar in (0.571, 1.0, 1.75):
    boxes, info = cp.plan(still, W, H, mode="static", crop_scale=1.5,
                          aspect_ratio=ar)
    check(boxes, still, f"aspect {ar}")
    got = boxes[0]["width"] / boxes[0]["height"]
    # both axes are on a 32 grid, so the ratio can only be approximate
    if abs(got - ar) / ar > 0.12:
        bad(f"aspect {ar}: got {got:.3f}, more than 12% off")

# --- a subject bigger than any legal box is REPORTED, not silently clipped --
# 720 is not a multiple of 32, so the tallest legal box is 704 and a subject
# filling the frame genuinely cannot be covered. The plan must say so.
huge = [(0, 0, W - 1, H - 1)] * 10
boxes, info = cp.plan(huge, W, H, mode="static", crop_scale=3.0)
if info["covered"]:
    bad("a subject taller than the tallest /32 box was reported as covered")
if boxes[0]["height"] != 704 or boxes[0]["width"] != 1280:
    bad(f"expected the largest legal box 1280x704, got {boxes[0]}")
# on a /32 image — which is what H3 Match Source Clip hands us — it fits
fits = [(0, 0, W - 1, 703)] * 10
boxes, info = cp.plan(fits, W, 704, mode="static", crop_scale=3.0)
if not info["covered"]:
    bad("a full-frame subject on a /32 image should be covered")

# --- tracked boxes are all one size (pixel-exact slicing depends on it) -----
boxes, _ = cp.plan(walk, W, H, mode="tracked", crop_scale=1.5)
if len({(b["width"], b["height"]) for b in boxes}) != 1:
    bad("tracked: box size must be constant across the clip")

if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all crop planner tests pass")
