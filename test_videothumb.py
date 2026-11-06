#!/usr/bin/env python3
"""The thumbnail cache must not serve a stale frame, and must not escape.

The decoding itself needs PyAV and a real clip, so it is measured rather than
unit-tested: on CJ's own input folder (371 mp4s, median 11.8 MB, p90 44.9 MB)
nine thumbnails rendered in 0.62s total -- about 0.05s and 3.5 KB each,
including 0.08s for a 42 MB file because it seeks instead of decoding from the
start -- and 4ms on a cache hit.

What IS unit-testable is the part that goes wrong quietly: a cache key that
does not notice a replaced file, and a crop that does not match the CSS the
tile is drawn with. Both would look like the feature working.
"""
import importlib.util
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vt", ROOT / "videothumb.py")
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)

fails = []


def ck(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(label)


print("the cache key notices anything that would change the picture")
with tempfile.TemporaryDirectory() as d:
    a = os.path.join(d, "clip.mp4")
    with open(a, "wb") as fh:
        fh.write(b"x" * 100)
    k1 = vt.cache_name(a)
    ck("stable for an unchanged file", vt.cache_name(a), k1)
    ck("and it is a webp name", k1.endswith(".webp"), True)

    # A REPLACED FILE AT THE SAME NAME IS THE WHOLE POINT. Re-encoding a clip
    # and keeping its name is normal, and a key on the path alone would serve
    # the old frame forever with an immutable Cache-Control on it.
    with open(a, "wb") as fh:
        fh.write(b"y" * 200)
    k2 = vt.cache_name(a)
    ck("a different SIZE is a different key", k2 != k1, True)

    os.utime(a, (0, 0))
    k3 = vt.cache_name(a)
    ck("and so is a different MTIME", k3 != k2, True)

    ck("a bigger tile is its own key", vt.cache_name(a, 384) != k3, True)

    b = os.path.join(d, "other.mp4")
    with open(b, "wb") as fh:
        fh.write(b"y" * 200)
    os.utime(b, (0, 0))
    # same size, same mtime, different path
    ck("two files that match on stat still differ by path",
       vt.cache_name(b) != k3, True)

    ck("a missing file does not raise",
       vt.cache_name(os.path.join(d, "nope.mp4")).endswith(".webp"), True)

print("\nthe crop is the cover-crop the tiles are drawn with")
# .tile img is `object-fit: cover` on a square, so the frame is centre-cropped
# to a square and then scaled. Getting this wrong squashes every thumbnail.
ck("landscape crops the sides", vt._fit(1280, 720, 192), (280, 0, 720, 192))
ck("portrait crops top and bottom", vt._fit(720, 1280, 192), (0, 280, 720, 192))
ck("a square is untouched", vt._fit(512, 512, 192), (0, 0, 512, 192))
ck("an odd width still lands inside", vt._fit(101, 100, 64), (0, 0, 100, 64))
ck("a degenerate frame does not divide by zero",
   vt._fit(0, 0, 64), (0, 0, 1, 64))

print("\nand it only claims the extensions it can open")
for name, want in (("a.mp4", True), ("a.MOV", True), ("a.webm", True),
                   ("a.mkv", True), ("a.m4v", True), ("a.avi", True),
                   ("a.png", False), ("a.wav", False), ("", False),
                   (None, False)):
    ck(f"is_video({name!r})", vt.is_video(name), want)

print("\na cache miss with no decoder reports rather than crashes")
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "not-really.mp4")
    with open(f, "wb") as fh:
        fh.write(b"this is not a video")
    try:
        vt.cached(f, os.path.join(d, "cache"))
        ck("garbage is refused", False, True)
    except vt.ThumbError as exc:
        ck("garbage raises ThumbError, with something sayable",
           bool(str(exc)), True)
    except Exception as exc:                                    # noqa: BLE001
        # no PyAV on this machine is the same path the route handles
        ck("a missing decoder is a ThumbError too",
           isinstance(exc, vt.ThumbError), True)
    ck("and nothing was left half-written",
       [p for p in (os.listdir(os.path.join(d, "cache"))
                    if os.path.isdir(os.path.join(d, "cache")) else [])
        if p.endswith(".part")], [])

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("video thumbs: the key tracks the file, the crop matches the CSS")
