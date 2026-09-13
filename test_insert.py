"""H3LatentInsert cuts a clip and generates NEW frames into the gap.

WHAT THIS FILE IS DEFENDING

  The node moves content. Everything else in this pack that moves content in
  latent space has been wrong at least once, because coverage is POSITIONAL:
  `FRAME_PER_TOKEN = (1,4,4,4,4)`, so latent step k covers 1 pixel frame when
  k%5==0 and 4 otherwise. Slide the tail by a number of steps that is not a
  multiple of 5 and every step lands on a coverage it was not encoded for --
  which renders as a clip that is subtly out of time rather than as an error.

  So the load-bearing check here is not "the mask holds the right region", it is
  "source step s came out at target step s + 5m, for every single step". The
  source is built with its own step index written into every cell, and the
  assertion reads the target back and demands the whole tail by value.

  The rest is the guards: a target canvas that is not source + insert, a hole
  with nothing in it, and the `x[..., -0:]` trap that made a zero-length tail
  hold the entire clip.

    python3 test_insert.py
"""
import sys

from _avstub import T, install, latent_t

mask = install()
from h3b.timing import audio_t  # noqa: E402

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)
        print(f"  FAIL {label}")


def stamped(t_v, h=4, w=4):
    """A video tensor whose every cell carries its own latent step index."""
    data = []
    for _ in range(24):
        for k in range(t_v):
            data.extend([float(k)] * (h * w))
    return T((1, 24, t_v, h, w), data=data)


def run(source_frames, insert, split, before=0, after=0, feather=0, strength=1.0,
        target_frames=None, wire_latent=True):
    target_frames = source_frames + insert if target_frames is None else target_frames
    s_v, t_v = latent_t(source_frames), latent_t(target_frames)
    s_a, t_a = audio_t(source_frames), audio_t(target_frames)
    src = [stamped(s_v), T((1, 32, 2, s_a), 9.0)]
    # -1.0 marks "came from the wired canvas"; a canvas the node allocates itself
    # is zeros, so the two are told apart by value in the assertions below.
    tgt = [T((1, 24, t_v, 4, 4), -1.0), T((1, 32, 2, t_a), 0.0)]
    node = mask.NODE_CLASS_MAPPINGS["H3LatentInsert"]()
    out = node.go({"samples": src}, split, insert, before, after, strength,
                  latent={"samples": tgt} if wire_latent else None,
                  audio_feather_ticks=feather)
    lat, gen, total, info = out["result"]
    v, _ = lat["samples"]
    return lat["samples"], lat["noise_mask"], gen, total, info, (
        s_v, int(v.shape[2]), s_a, audio_t(total))


SRC = 345                  # 17*20+5 -> 102 video steps, 575 audio ticks
INS = 68                   # 4 VAE chunks -> 20 video steps

print("an interior insert moves the tail by whole groups, and by value")
(sv, sa), (mv, ma), gen, total, info, (s_v, t_v, s_a, t_a) = run(SRC, INS, 170)
check("the source is 102 steps and the target 122", (s_v, t_v), (102, 122))
check("the take is 68 frames longer", total, SRC + INS)
check("split 170 is 10 chunks, so 50 steps stay at the head",
      [mv.at(k, axis=2)[0] for k in (0, 49, 50)], [0.0, 0.0, 1.0])
check("the head is the source's own steps, unmoved",
      [sv.at(k, axis=2)[0] for k in (0, 49)], [0.0, 49.0])
# THE CHECK THIS FILE EXISTS FOR. 20 inserted steps, so source step s has to
# reappear at target step s+20 -- every one of them, not just the boundary.
moved = [(k, sv.at(k, axis=2)[0]) for k in range(70, t_v)]
check("every tail step lands exactly 20 steps later",
      [k - v for k, v in moved], [20] * len(moved))
check("the tail reaches the very last step", sv.at(t_v - 1, axis=2)[0], float(s_v - 1))
check("and the whole tail is held",
      [mv.at(k, axis=2)[0] for k in (69, 70, t_v - 1)], [1.0, 0.0, 0.0])
check("the gap is untouched target", sv.at(60, axis=2)[0], -1.0)
check("generated_frames is the inserted span", gen, INS)

print("audio is anchored at both ends, so no rounding accumulates")
check("the head holds audio_t(170) ticks",
      [ma.at(k)[0] for k in (0, audio_t(170) - 1, audio_t(170))], [0.0, 0.0, 1.0])
check("the tail holds the rest, ending at the end",
      [ma.at(k)[0] for k in (t_a - (s_a - audio_t(170)) - 1,
                             t_a - (s_a - audio_t(170)), t_a - 1)], [1.0, 0.0, 0.0])
check("the generated gap is exactly the insert's duration",
      t_a - s_a, (t_a - (s_a - audio_t(170))) - audio_t(170))
check("held audio carries the source", sa.at(0)[0], 9.0)

print("split 0 puts the new frames in front, split at the end puts them after")
(svh, _), (mvh, _), genh, _, _, (_, t_vh, _, _) = run(SRC, INS, 0)
check("nothing is held at the head", mvh.at(0, axis=2)[0], 1.0)
check("the whole source sits at the end, shifted 20",
      [svh.at(k, axis=2)[0] for k in (20, t_vh - 1)], [0.0, float(s_v - 1)])
check("the first 20 steps are the new ones", mvh.at(19, axis=2)[0], 1.0)
check("and step 20 begins the held source", mvh.at(20, axis=2)[0], 0.0)
check("a prepend generates the insert", genh, INS)

(svt, _), (mvt, _), gent, _, _, (_, t_vt, _, _) = run(SRC, INS, SRC)
check("the whole source stays where it was",
      [svt.at(k, axis=2)[0] for k in (0, s_v - 1)], [0.0, float(s_v - 1)])
check("everything past it is generated",
      [mvt.at(k, axis=2)[0] for k in (s_v - 1, s_v, t_vt - 1)], [0.0, 1.0, 1.0])
check("an append generates the insert", gent, INS)
# a split past the last frame is an append, not an error -- the last group is 5
# frames, so 340 and 345 both mean "the end"
check("a split inside the final 5-frame group still appends",
      run(SRC, INS, 344)[1][0].at(s_v, axis=2)[0], 1.0)

print("blend_before / blend_after widen the hole into the source")
(svb, _), (mvb, _), genb, _, _, (_, t_vb, _, _) = run(SRC, INS, 170, before=34, after=51)
check("the head now stops at 136 frames = 40 steps",
      [mvb.at(k, axis=2)[0] for k in (39, 40)], [0.0, 1.0])
check("the tail now starts at source frame 221 -> 65 steps, +20",
      mvb.at(85, axis=2)[0], 0.0)
check("the step before it is generated", mvb.at(84, axis=2)[0], 1.0)
check("the tail is still phase-correct", svb.at(85, axis=2)[0], 65.0)
check("generated_frames counts the run-up and run-out too", genb, INS + 34 + 51)
# asked for 18 frames of run-out and got exactly 18: frame 188 is a step
# boundary (11*17+1), which is source step 56, which is target step 76. The
# coarse version rounded this to 34 frames -- the whole point of the finer grid.
check("blend_after snaps UP to the nearest step, never less than asked",
      [run(SRC, INS, 170, after=18)[1][0].at(k, axis=2)[0] for k in (75, 76)],
      [1.0, 0.0])

print("insert_frames 0 is the bracket: a hole, same length")
(_, _), (mv0, _), gen0, total0, _, (_, t_v0, _, _) = run(SRC, 0, 170, before=34, after=34)
check("the take is not longer", total0, SRC)
check("and a hole is open in the middle",
      [mv0.at(k, axis=2)[0] for k in (39, 40, 59, 60)], [0.0, 1.0, 1.0, 0.0])
check("generated_frames is just the blend", gen0, 68)

print("the audio feather runs the right way at each end")
(_, _), (_, maf), _, _, _, (_, _, _, t_af) = run(SRC, INS, 170, feather=8)
h = audio_t(170)
ok("the head ramp rises into the gap",
   maf.at(h - 8)[0] < maf.at(h - 4)[0] < maf.at(h - 1)[0] < 1.0)
ok("the tail ramp falls into the held tail",
   maf.at(t_af - 292)[0] > maf.at(t_af - 292 + 4)[0] > maf.at(t_af - 292 + 7)[0])
check("deep inside the head is still fully held", maf.at(0)[0], 0.0)
check("deep inside the tail is still fully held", maf.at(t_af - 1)[0], 0.0)

print("the cut lands on a latent step, which is 4x finer than a VAE chunk")
# THE 17 BELONGS TO THE SHIFT, NOT THE CUT. Five steps cover 17 frames wherever
# they start, so the tail moves a whole group whichever boundary it begins on.
# The first version put the cut on the 17s too and threw away 4x the precision.
check("a group's steps start at 0, 1, 5, 9, 13",
      [mask.frames_at_step(k) for k in range(6)], [0, 1, 5, 9, 13, 17])
check("and the pattern repeats every 17 frames",
      [mask.frames_at_step(k) for k in (5, 6, 7, 8, 9, 10)],
      [17, 18, 22, 26, 30, 34])
check("frames snap DOWN to the boundary at or below",
      [mask.step_at_frame(f) for f in (0, 1, 4, 5, 12, 13, 16, 17)],
      [0, 1, 1, 2, 3, 4, 4, 5])
check("or UP, when holding less than asked is the wrong way to round",
      [mask.step_at_frame(f, "up") for f in (0, 1, 2, 5, 14, 17, 18)],
      [0, 1, 2, 2, 5, 5, 6])
for f in range(0, 200):
    got = mask.frames_at_step(mask.step_at_frame(f))
    if got > f or mask.frames_at_step(mask.step_at_frame(f) + 1) <= f:
        fails.append(f"snap-down is not the nearest boundary at frame {f}")
        print(f"  FAIL snap-down at {f}: got {got}")
        break

# a cut at 103 -- one past a chunk boundary, so the old grid would have thrown it
# back to 102. 103 sits between boundaries 102 and 103... frame 103 IS 6*17+1.
(sv4, _), (mv4, _), gen4, _, info4, (_, t_v4, _, _) = run(SRC, INS, 103)
check("cutting at 103 holds 31 steps, not 30", mv4.at(30, axis=2)[0], 0.0)
check("and step 31 is generated", mv4.at(31, axis=2)[0], 1.0)
check("the tail still lands 20 steps later", sv4.at(t_v4 - 1, axis=2)[0],
      float(latent_t(SRC) - 1))
check("the inserted span is still exactly 68 frames", gen4, INS)
ok("an off-boundary request says where it actually landed",
   "asked for 104" in run(SRC, INS, 104)[4])

print("it sizes its own canvas, so no upstream node has to be told the length")
# THE DESIGN THIS REPLACED asked for a canvas of source+insert and refused
# anything else, which meant the same number typed into two nodes. It failed on
# its first real render for exactly that reason. The length lives only in the
# latent, so the node can work it out -- and a wired canvas of the wrong length
# is now REPLACED and reported, not refused.
for label, kwargs in (
        ("nothing wired at all", dict(wire_latent=False)),
        ("a canvas that was never lengthened", dict(target_frames=SRC)),
        ("a canvas lengthened by the wrong amount", dict(target_frames=SRC + 34))):
    args = dict(source_frames=SRC, insert=INS, split=170)
    args.update(kwargs)
    (sv2, _), (mv2, _), gen2, total2, info2, (_, t_v2, _, _) = run(**args)
    check(f"{label}: the take is still {SRC + INS}", total2, SRC + INS)
    check(f"{label}: and the canvas is {latent_t(SRC + INS)} steps",
          t_v2, latent_t(SRC + INS))
    check(f"{label}: the tail still lands 20 steps later",
          sv2.at(t_v2 - 1, axis=2)[0], float(latent_t(SRC) - 1))
    ok(f"{label}: the generated gap is fresh canvas, not the wired one",
       sv2.at(60, axis=2)[0] == 0.0)
ok("a replaced canvas says so in the info",
   "canvas was built here instead" in run(SRC, INS, 170, target_frames=SRC)[4])
ok("a right-sized wired canvas is used as given",
   run(SRC, INS, 170)[0][0].at(60, axis=2)[0] == -1.0)

# THE RENDER THAT FAILED, 2026-09-12: 192 frames, cut at 51, 187 inserted. It
# failed because the canvas had been sized 243 by a second widget on another
# node. Nothing upstream is told anything now, so it just works.
print("the render this was rebuilt for")
(sv3, _), (mv3, _), gen3, total3, _, (_, t_v3, _, _) = run(192, 187, 51,
                                                           wire_latent=False)
check("192 + 187 is 379, which is 17*22+5", total3, 379)
check("and the canvas is sized for it", t_v3, latent_t(379))
check("51 frames = 15 steps stay at the head",
      [mv3.at(k, axis=2)[0] for k in (14, 15)], [0.0, 1.0])
check("the source's tail lands at the end",
      sv3.at(t_v3 - 1, axis=2)[0], float(latent_t(192) - 1))
check("187 frames are generated", gen3, 187)

print("it still refuses a hole with nothing in it")
try:
    run(SRC, 0, 170)
    fails.append("a hole with nothing in it: no error raised")
    print("  FAIL a hole with nothing in it: no error raised")
except ValueError:
    pass

print("the info says where the new frames went")
ok("an interior insert names the frame", "at frame 170" in info)
ok("it reports both lengths", "345 -> 413" in info)
ok("a prepend says so", "in front of the clip" in run(SRC, INS, 0)[4])
ok("an append says so", "after the clip" in run(SRC, INS, SRC)[4])

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("insert: the tail moves by whole groups, by value, and the canvas is checked")
