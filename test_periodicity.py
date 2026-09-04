"""H3 Periodicity — does it find a planted seam, and stay quiet without one?

THE QUESTION IT ANSWERS
  H3's VAE slices every 17 frames in `encode_temporal`, disjointly, with no
  overlap and no blend. spacepxl called that a defect on 2026-09-03. It may not
  be: 17 is the model's native block, so slicing there may be architecture
  rather than a seam. Nobody has measured it.

WHY THE CONTROL PERIODS ARE THE WHOLE TEST
  Any periodicity statistic on real footage finds something, because motion is
  not stationary. A ratio above 1 at period 17 is worthless on its own. What
  makes it evidence is 17 standing clear of 16, 18 and 19 -- periods with no
  architectural meaning. These tests check both halves: that a planted seam is
  found AND that ordinary footage does not produce a false one.

  Synthetic series only. No torch, no ComfyUI, no render.

    python3 test_periodicity.py
"""
import importlib.util
import math
import pathlib
import random
import sys

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location("h3per", _root / "periodicity.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


# --- folding ---------------------------------------------------------------- #
print("phase 0 is the BOUNDARY: series[i] is the step from frame i to i+1")
# a spike between frame 17 and 18 lives at series index 17
s = [0.1] * 60
s[17 - 1] = 0.9          # the step INTO frame 17
ph = pr.fold(s, 17)
check("a step into frame 17 folds to phase 0",
      max(range(17), key=lambda k: ph[k]), 0)

print("offset shifts which absolute frame is phase 0")
ph2 = pr.fold(s, 17, offset=3)
ok("with offset 3 the peak moves off phase 0",
   max(range(17), key=lambda k: ph2[k]) != 0)

# --- a planted seam is found ------------------------------------------------ #
print("a planted 17-frame seam is found, and beats every control period")
random.seed(4)
base = [0.05 + random.random() * 0.01 for _ in range(17 * 30)]
seam = list(base)
for i in range(len(seam)):
    if (i + 1) % 17 == 0:
        seam[i] *= 3.0
r17, peak, _ = pr.score(seam, 17)
ok(f"period 17 ratio is well above 1 ({r17:.2f}x)", r17 > 2.0)
check("and the strongest phase IS the boundary", peak, 0)
ctrl = [pr.score(seam, p)[0] for p in (16, 18, 19, 23)]
ok(f"it beats every control {['%.2f' % c for c in ctrl]}",
   all(r17 > c for c in ctrl))

# --- ordinary footage does not manufacture one ------------------------------ #
print("smooth motion with no seam produces no standout at 17")
random.seed(9)
# a slow ramp plus noise: lumpy, non-stationary, but nothing periodic at 17
mot = [0.05 + 0.03 * math.sin(i / 40.0) + random.random() * 0.01
       for i in range(17 * 30)]
r_no, _, _ = pr.score(mot, 17)
ctrl_no = [pr.score(mot, p)[0] for p in (16, 18, 19, 23)]
ok(f"17 is not separated from its neighbours "
   f"({r_no:.3f} vs max control {max(ctrl_no):.3f})",
   r_no <= max(ctrl_no) + 0.05)
ok("and the ratio is near 1", abs(r_no - 1.0) < 0.25)

print("a seam at a DIFFERENT period does not read as a 17-frame effect")
random.seed(11)
other = [0.05 + random.random() * 0.01 for _ in range(23 * 30)]
for i in range(len(other)):
    if (i + 1) % 23 == 0:
        other[i] *= 3.0
r23 = pr.score(other, 23)[0]
r17b = pr.score(other, 17)[0]
ok(f"23 fires ({r23:.2f}x) and 17 does not ({r17b:.2f}x)",
   r23 > 2.0 and r17b < 1.5)

# --- the node's own guards -------------------------------------------------- #
print("too short to have cycles is reported, not answered")
class Img:
    def __init__(self, n): self.n = n
    @property
    def shape(self): return (self.n, 64, 64, 3)
res = pr.H3Periodicity().go(Img(20), 17, 0)
txt = res["result"][0]
ok("says it is too short", "too short" in txt)
check("and returns a neutral ratio", res["result"][1], 1.0)

print("the default period is the VAE's clip length, not a guess")
check("clip_length", pr.VAE_CLIP_LENGTH, 17)
t = pr.H3Periodicity.INPUT_TYPES()
check("default period", t["required"]["period"][1]["default"], 17)
ok("controls default to nearby meaningless periods",
   "16" in t["optional"]["controls"][1]["default"])
check("returns", pr.H3Periodicity.RETURN_NAMES, ("info", "ratio"))

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("periodicity: all checks pass")
