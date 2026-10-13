#!/usr/bin/env python3
"""The export, RUN — not just asserted on.

A filter graph is wrong in ways that produce a file rather than an error: a
clip a frame late, an overlay that smears its last frame over everything
after it, a gap that freezes instead of going black, a stack composited
upside down. None of those raise. So this builds real clips of known colours,
runs the real command, and reads the colours back out of the result.

Skips cleanly when ffmpeg is absent — it is a check on the graph, not a
dependency of the pack.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import editexport as EX                                        # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not (FFMPEG and FFPROBE):
    print("export: SKIPPED — no ffmpeg on PATH")
    sys.exit(0)

fails = []


def ok(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


def check(label, got, want):
    ok(f"{label}: {got!r}", got == want)


tmp = pathlib.Path(tempfile.mkdtemp())


def make(colour, secs=4, audio=True):
    p = tmp / f"{colour}.mp4"
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c={colour}:s=160x240:r=24:d={secs}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(p)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(p)


def colour_at(path, t):
    """The dominant RGB of one frame, as a coarse name."""
    out = tmp / "f.png"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(t), "-i", path, "-frames:v", "1",
                    "-vf", "scale=1:1", str(out)],
                   check=True, capture_output=True)
    raw = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(out),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    r, g, b = raw[0], raw[1], raw[2]
    if max(r, g, b) < 40:
        return "black"
    return {max((r, "red"), (g, "green"), (b, "blue"))[1]: 1}.popitem()[0]


def probe(path, key):
    return subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", key, "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True).stdout.strip()


red, green, blue = make("red"), make("green"), make("blue")
print("built three 4s clips: red, green, blue\n")

# ---------------------------------------------------- a cut, and a gap
print("a sequence with a GAP between two clips")
proj = {"tracks": [{"kind": "video", "clips": [
    {"source": red, "in": 0, "out": 48, "start": 0},        # 0-2s
    {"source": blue, "in": 0, "out": 48, "start": 96},      # 4-6s
]}]}
out = str(tmp / "gap.mp4")
argv, notes = EX.build(proj, out, 160, 240, ffmpeg=FFMPEG)
r = subprocess.run(argv, capture_output=True, text=True)
ok(f"it runs — {r.returncode}", r.returncode == 0)
if r.returncode:
    print(r.stderr[-700:])
check("the result is as long as the timeline, not the clips",
      round(float(probe(out, "format=duration"))), 6)
check("the first clip plays", colour_at(out, 1), "red")
check("THE GAP IS BLACK, not a frozen frame", colour_at(out, 3), "black")
check("and the second clip plays at ITS start", colour_at(out, 5), "blue")

# ------------------------------------------------------------- layering
print("\nlayers: a clip on V2 covers the bed on V1")
# the bed is 96 frames because red.mp4 IS 96 frames. Asking for 144 made the
# first version of this test fail at t=5 with "black" — which was correct,
# there was no red left. A clip cannot be longer than its source, and the
# test has to respect that or it measures its own mistake.
proj = {"tracks": [
    {"kind": "video", "clips": [{"source": green, "in": 0, "out": 48,
                                 "start": 24}]},            # V2, 1-3s
    {"kind": "video", "clips": [{"source": red, "in": 0, "out": 96,
                                 "start": 0}]},             # V1, 0-4s
]}
out2 = str(tmp / "layer.mp4")
argv, _ = EX.build(proj, out2, 160, 240, ffmpeg=FFMPEG)
r = subprocess.run(argv, capture_output=True, text=True)
ok(f"it runs — {r.returncode}", r.returncode == 0)
if r.returncode:
    print(r.stderr[-700:])
check("before the overlay, the bed shows", colour_at(out2, 0.5), "red")
check("THE TOP TRACK WINS where they overlap", colour_at(out2, 2), "green")
check("and the bed RETURNS after the overlay ends — the overlay does not "
      "smear its last frame forward", colour_at(out2, 3.5), "red")

# -------------------------------------------------------------- trimming
print("\na trim exports the trimmed part, at the trimmed length")
proj = {"tracks": [{"kind": "video", "clips": [
    {"source": blue, "in": 24, "out": 72, "start": 0},       # 1s-3s of blue
]}]}
out3 = str(tmp / "trim.mp4")
argv, _ = EX.build(proj, out3, 160, 240, ffmpeg=FFMPEG)
r = subprocess.run(argv, capture_output=True, text=True)
ok(f"it runs — {r.returncode}", r.returncode == 0)
check("two seconds, not four", round(float(probe(out3, "format=duration"))), 2)
check("and it is the right clip", colour_at(out3, 1), "blue")

# ----------------------------------------------------------------- audio
print("\naudio comes through, and an empty timeline is refused")
ok("the mix is present", bool(probe(out, "stream=codec_type").count("audio")))
try:
    EX.build({"tracks": []}, "x.mp4", 160, 240)
    ok("an empty project raises", False)
except EX.ExportError as e:
    ok(f"an empty project raises — {e}", True)

print("\nand the command is deterministic")
a1, _ = EX.build(proj, out3, 160, 240, ffmpeg=FFMPEG)
a2, _ = EX.build(proj, out3, 160, 240, ffmpeg=FFMPEG)
ok("built twice, identical", a1 == a2)

shutil.rmtree(tmp, ignore_errors=True)
print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("export: gaps go black, the top track wins, trims are honoured")
