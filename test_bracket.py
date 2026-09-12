"""H3LatentBracket holds both ends of a clip and generates the middle.

WHAT THIS FILE IS DEFENDING

  The node's whole contract is the noise mask it builds, and every way of
  getting that wrong is SILENT. A mask that holds too much renders a clip that
  ignores the prompt; a mask that holds too little renders one that ignores the
  source; both come back as "the model did something odd" rather than an error.

  The specific trap, and the reason this file exists: `x[..., -0:]` is the WHOLE
  tensor in Python, not an empty slice. A zero-length tail bracket written the
  obvious way holds the entire clip and generates nothing at all -- a render
  that costs full price and returns its own input.

  The arithmetic is checked against `timing`'s own helpers rather than repeated
  here, so a change to the frame grid fails in one place.

    python3 test_bracket.py
"""
import sys

from _avstub import T, install, latent_t

mask = install()
from h3b.timing import audio_t, frame_groups  # noqa: E402

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)
        print(f"  FAIL {label}")


def run(total_frames, head, tail, feather=0, strength=1.0):
    t_v = latent_t(total_frames)
    t_a = audio_t(total_frames)
    src = [T((1, 24, t_v, 4, 4), 7.0), T((1, 32, 2, t_a), 9.0)]
    tgt = [T((1, 24, t_v, 4, 4), 0.0), T((1, 32, 2, t_a), 0.0)]
    node = mask.NODE_CLASS_MAPPINGS["H3LatentBracket"]()
    out = node.go({"samples": tgt}, {"samples": src}, head, tail, strength,
                  audio_feather_ticks=feather)
    lat = out["result"][0]
    return lat["samples"], lat["noise_mask"], out["result"][1], t_v, t_a


TOTAL = 345                      # 17*20+5, 102 video steps, 575 audio ticks
print("both ends are held and the middle is left to generate")
(sv, sa), (mv, ma), gen, t_v, t_a = run(TOTAL, 34, 34)
check("the clip is 102 video steps", t_v, 102)
check("34 frames is 10 video steps at each end",
      (mv.at(0, axis=2)[0], mv.at(9, axis=2)[0]), (0.0, 0.0))
check("step 10 is generated", mv.at(10, axis=2)[0], 1.0)
check("the last 10 steps are held",
      (mv.at(t_v - 10, axis=2)[0], mv.at(t_v - 1, axis=2)[0]), (0.0, 0.0))
check("the step before them is generated", mv.at(t_v - 11, axis=2)[0], 1.0)
check("held video carries the SOURCE, not the target",
      (sv.at(0, axis=2)[0], sv.at(t_v - 1, axis=2)[0]), (7.0, 7.0))
check("the middle is untouched target", sv.at(50, axis=2)[0], 0.0)
check("generated_frames is what is left", gen, sum(frame_groups(t_v)) - 68)

print("audio is held over the same real time")
check("34 frames is 57 ticks", audio_t(34), 57)
check("the first ticks are held", ma.at(0)[0], 0.0)
check("and the last", ma.at(t_a - 1)[0], 0.0)
check("the middle is generated", ma.at(t_a // 2)[0], 1.0)
check("held audio carries the source", sa.at(0)[0], 9.0)

# THE BUG THIS FILE WAS WRITTEN FOR. `x[..., -0:]` is the whole tensor, so a
# zero tail written the obvious way holds everything and generates nothing.
print("a zero-length bracket holds nothing, rather than everything")
(_, _), (mv0, ma0), _, t_v0, t_a0 = run(TOTAL, 34, 0)
check("the tail is free when tail_frames is 0", mv0.at(t_v0 - 1, axis=2)[0], 1.0)
check("and its audio is too", ma0.at(t_a0 - 1)[0], 1.0)
check("the head is still held", mv0.at(0, axis=2)[0], 0.0)
(_, _), (mv1, _), _, t_v1, _ = run(TOTAL, 0, 34)
check("the head is free when head_frames is 0", mv1.at(0, axis=2)[0], 1.0)
check("the tail is still held", mv1.at(t_v1 - 1, axis=2)[0], 0.0)

print("brackets snap to whole VAE chunks")
# 5 latent steps cover 17 pixel frames wherever they start, so a held region is
# always a round number of chunks and needs no phase arithmetic.
(_, _), (mv2, _), _, t_v2, _ = run(TOTAL, 40, 0)      # 40 -> 34
check("40 frames snaps down to 34, which is 10 steps",
      (mv2.at(9, axis=2)[0], mv2.at(10, axis=2)[0]), (0.0, 1.0))
(_, _), (mv3, _), _, _, _ = run(TOTAL, 16, 0)         # 16 -> 0
check("below one chunk nothing is held", mv3.at(0, axis=2)[0], 1.0)

print("the audio feather runs the right way at each end")
(_, _), (_, maf), _, _, t_af = run(TOTAL, 51, 51, feather=8)
h = audio_t(51)
ok("the head ramp rises into the generated middle",
   maf.at(h - 8)[0] < maf.at(h - 4)[0] < maf.at(h - 1)[0] < 1.0)
ok("the tail ramp falls into the held tail",
   maf.at(t_af - h)[0] > maf.at(t_af - h + 4)[0] > maf.at(t_af - h + 7)[0])
check("deep inside the head is still fully held", maf.at(0)[0], 0.0)
check("deep inside the tail is still fully held", maf.at(t_af - 1)[0], 0.0)
check("a feather of 0 leaves a hard edge",
      run(TOTAL, 51, 51, feather=0)[1][1].at(h - 1)[0], 0.0)

print("it refuses what it cannot do")
for label, args in (("holding the whole clip", (TOTAL, 340, 340)),
                    ("a head longer than the clip", (TOTAL, 3400, 0))):
    try:
        run(*args)
        fails.append(f"{label}: no error raised")
        print(f"  FAIL {label}: no error raised")
    except ValueError:
        pass

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("bracket: both ends held, the middle generated, and -0 does not mean all")
