"""chunk_count in, duration out — the arithmetic a front end previews with.

WHAT THIS DEFENDS

  The node and the /chunk/plan HTTP route both answer "how long will this be".
  Two implementations of one rule is this pack's recurring failure, so both call
  `count_plan` and this pins what it returns.

  The load-bearing fact is the ASYMMETRY. Chunk 1 keeps everything it renders;
  every later chunk drops the carried handle. Those two lengths can never be
  equal -- it needs A = B - C with all three on the 17n+5 grid, and every legal
  run is 5 mod 17, so B-C is 0 mod 17 while A is 5. A UI that assumes equal
  segments will mis-state the duration to a user, and nothing will error.

    python3 test_countplan.py
"""
import sys
import types

sys.modules.setdefault("torch", types.ModuleType("torch"))
import chunkplan as cp  # noqa: E402

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def ok(label, cond):
    check(label, bool(cond), True)


print("the total follows from the count, and inverts against the planner")
for cf, ctx, n in ((192, 39, 9), (141, 39, 4), (124, 39, 4), (90, 22, 6)):
    total = cp.total_for_count(cf, n, ctx)
    chunks, _ = cp.plan(total, cf, "fixed", context=ctx, grow_tail=True)
    check(f"{n}x{cf} ctx{ctx} -> {total} frames gives back {n} chunks",
          len(chunks), n)

check("the confirmed 9x192/39 take is 1416 frames",
      cp.total_for_count(192, 9, 39), 1416)
check("and 59.00 seconds", round(1416 / 24, 2), 59.0)

print("the asymmetry is reported, because it cannot be removed")
total, first, rest, notes = cp.count_plan(141, 4, 39)
check("chunk 1 delivers the whole chunk", first, 141)
check("later chunks drop the carry", rest, 102)
check("total is first + 3 x rest", total, 141 + 3 * 102)
# EXHAUSTIVE, not a spot check: if a solution existed the UI could promise even
# segments, and it cannot.
sols = [(a, b, c) for a in cp.LEGAL_RUNS for b in cp.LEGAL_RUNS
        for c in cp.LEGAL_RUNS if a == b - c]
check("equal delivered segments are impossible on the legal grid", sols, [])

print("illegal and off-grid inputs are named, not silently fixed")
ok("a non-legal chunk size is called out",
   any("not a legal run" in x for x in cp.count_plan(120, 4, 39)[3]))
ok("a non-legal context is called out",
   any("context 40" in x for x in cp.count_plan(141, 4, 40)[3]))
ok("a context that leaves nothing to generate is called out",
   any("nothing new" in x for x in cp.count_plan(141, 4, 141)[3]))
ok("an off-audio-grid chunk size is called out",
   any("40 Hz grid" in x for x in cp.count_plan(124, 4, 39)[3]))
ok("and an audio-exact one is not",
   not any("40 Hz grid" in x for x in cp.count_plan(141, 4, 39)[3]))
check("the audio-exact runs are 39, 90, 141, 192, 243 ...",
      [r for r in cp.AV_EXACT_RUNS if r <= 243], [39, 90, 141, 192, 243])

print("the report answers the two questions an author actually has")
text = cp.describe_count_plan(141, 4, 39)
ok("how long will it be", "18.62s" in text)
ok("how many beats do I write", "write 4 beat(s)" in text)
ok("and it offers nearby sizes", "alternative:" in text)

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("count plan: duration follows the count, and the asymmetry is stated")


print("the audio grid is stated where the choice is made")
import importlib.util as _il, pathlib as _pl, types as _t
for _m in ("torch.nn", "torch.nn.functional", "comfy", "comfy.utils", "folder_paths"):
    sys.modules.setdefault(_m, _t.ModuleType(_m))
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
_root = _pl.Path(__file__).resolve().parent
_pkg = _t.ModuleType("h3cp"); _pkg.__path__ = [str(_root)]; sys.modules["h3cp"] = _pkg
def _load(n):
    sp = _il.spec_from_file_location(f"h3cp.{n}", _root / f"{n}.py")
    m = _il.module_from_spec(sp); sys.modules[f"h3cp.{n}"] = m
    sp.loader.exec_module(m); return m
for _n in ("timing", "chunkplan", "avlatent", "geometry", "cropplan", "crop"):
    try: _load(_n)
    except Exception: pass
lf = _load("longform")
PLAN = lf.NODE_CLASS_MAPPINGS["H3ChunkPlan"]
spec = PLAN.INPUT_TYPES()

# 22 IS THE ONLY CARRY OFF BOTH CLOCKS (36.67 ticks) and it was the default for
# months, so anything that did not override it inherited the worst option.
ctx = spec["optional"]["context"]
check("context defaults to 39, the smallest carry on both clocks",
      ctx[1]["default"], "39")
ok("39 is first in the list", ctx[0][0] == "39")
ok("39 really is audio-exact", cp.count_plan(141, 2, 39)[3] == [])

# chunk_frames STAYS AN INT: seven saved graphs drive it from a link, and an INT
# link into a COMBO is invalid. A dropdown would have put the audio grid at the
# point of choosing and made the value unwireable -- tried, reverted, and the
# tooltip carries the grid instead.
cf = spec["required"]["chunk_frames"]
check("chunk_frames is an INT so it can be driven by a link", cf[0], "INT")
check("stepping by 17 keeps the arrows on legal runs", cf[1]["step"], 17)
check("and it defaults to the nearest audio-exact size to 5s", cf[1]["default"], 141)
ok("the tooltip names the audio-exact sizes", "40 Hz AUDIO grid" in cf[1]["tooltip"])

print("av_aligned snaps UP and says what it cost")
r = PLAN().go(chunk_frames=158, chunk_mode="fixed",
              chunk_count=4, context="39", av_aligned=True)
ok("it reports the snap", "snapped UP to 192" in r["result"][2])
ok("and the price per chunk", "Costs 34 frame(s)" in r["result"][2])
check("the plan uses the snapped size", r["result"][3],
      cp.total_for_count(192, 4, 39))
r = PLAN().go(chunk_frames=141, chunk_mode="fixed",
              chunk_count=4, context="39", av_aligned=True)
ok("an already-aligned size is left alone", "snapped" not in r["result"][2])

if fails:
    print(f"\nFAIL — {len(fails)} check(s)")
    sys.exit(1)
print("\naudio grid: stated at the choice, defaulted sanely, snapped on request")


print("continuity: the transition decides the pin")
check("carry pins the context", cp.continuity_pin("carry", 39), (39, "carry", True))
check("cut pins nothing", cp.continuity_pin("cut", 39), (0, "fresh", True))
check("handoff pins one frame", cp.continuity_pin("handoff", 39), (1, "fresh", True))
# THE ONE OUR OWN NOTES ASKED FOR: generated audio restarts at every link, so a
# picture cut takes the music with it unless the two decisions are separable.
check("audio_carry cuts the picture and keeps the sound",
      cp.continuity_pin("audio_carry", 39), (0, "carry", True))
ok("an unknown name falls back to carry rather than erroring",
   cp.continuity_pin("nonsense", 39) == cp.continuity_pin("carry", 39))
ok("every kind says what it does", all("what" in v for v in cp.CONTINUITY.values()))
ok("every UNWIRED kind says why", all("why" in v for v in cp.CONTINUITY.values()
                                      if not v["ok"]))

print("and a plan carries it per chunk")
_t = cp.total_for_count(192, 4, 39)
_ch, _info = cp.plan(_t, 192, "fixed", context=39, grow_tail=True,
                     continuity=["carry", "cut", "audio_carry"])
check("pins follow the transitions", [c["pin"] for c in _ch], [0, 39, 0, 0])
check("and so does the soundtrack",
      [c["carry_audio"] for c in _ch], [False, True, False, True])
check("chunk 1 has no transition into it", _ch[0]["continuity"], None)
# reference_sample is WIRED now (H3ChunkRefSample), so it is no longer the
# example of an unnamed-but-unbuilt kind. `refresh` still is: it needs a
# fractional denoise mask on the carried rows, which 0.35+ turns into rows that
# never finish denoising.
_ch2, _info2 = cp.plan(_t, 192, "fixed", context=39, grow_tail=True,
                       continuity=["refresh"])
ok("an unwired kind is reported, not silently swapped",
   any("NOT WIRED" in n for n in _info2["notes"]))
_ch3b, _info3b = cp.plan(_t, 192, "fixed", context=39, grow_tail=True,
                         continuity=["reference_sample"])
ok("reference_sample no longer warns, because it is wired",
   not any("NOT WIRED" in n for n in _info3b["notes"]))
check("and it plans as a cut for the picture", _ch3b[1]["pin"], 0)
_ch3, _ = cp.plan(_t, 192, "fixed", context=39, grow_tail=True)
check("no continuity at all behaves exactly as before",
      [c["pin"] for c in _ch3], [0, 39, 39, 39])

print("\nscene_plan: every scene sets its own length")
# count_plan answers "n chunks of one size", which is a continuous take. A
# four-second insert and a twelve-second master belong in the same film, and
# that needed a different planner rather than a bigger chunk_count.
_sp = cp.scene_plan([
    {"frames": 192, "beat": "master"},
    {"frames": 90, "continuity": "carry", "beat": "she turns"},
    {"frames": 141, "continuity": "cut", "beat": "new angle"},
    {"frames": 56, "continuity": "audio_carry", "beat": "insert"},
], context=39)
check("each scene renders its own length",
      [c["frames"] for c in _sp["scenes"]], [192, 90, 141, 56])
check("a carry overlaps and delivers less",
      [c["pin"] for c in _sp["scenes"]], [0, 39, 0, 0])
check("so delivered is frames minus the pin",
      [c["delivered"] for c in _sp["scenes"]], [192, 51, 141, 56])
check("and the timeline is contiguous",
      [(c["start"], c["end"]) for c in _sp["scenes"]],
      [(0, 192), (192, 243), (243, 384), (384, 440)])
check("total is the sum of what reaches the clip", _sp["total_frames"], 440)
check("scene 1 has no transition in", _sp["scenes"][0]["continuity"], None)
check("audio_carry keeps the sound across its cut",
      [c["carry_audio"] for c in _sp["scenes"]], [False, True, False, True])

print("a scene may override the carry, and says when it did")
# the global `context` is right most of the time, but a scene continuing from a
# fast-moving shot has no reason to pin as much as one continuing from a held
# frame — so the global is a DEFAULT, not the only answer
_ov = cp.scene_plan([{"frames": 192},
                     {"frames": 192, "continuity": "carry"},
                     {"frames": 192, "continuity": "carry", "pin": 5}],
                    context=39)
check("the default applies where nothing is set",
      _ov["scenes"][1]["pin"], 39)
check("and an override wins", _ov["scenes"][2]["pin"], 5)
check("so that scene delivers more",
      _ov["scenes"][2]["delivered"], 187)
check("the default is reported either way",
      [c["pin_default"] for c in _ov["scenes"]], [39, 39, 39])
check("and which scenes overrode it",
      [c["pin_overridden"] for c in _ov["scenes"]], [False, False, True])
ok("a pin of 0 is an override, not an absence",
   cp.scene_plan([{"frames": 192}, {"frames": 192, "continuity": "carry",
                                    "pin": 0}], context=39)["scenes"][1]["pin"] == 0)

print("and it refuses to lie about illegal lengths")
_sp2 = cp.scene_plan([{"frames": 100}], context=39)
check("100 is snapped to a legal run", _sp2["scenes"][0]["frames"], 107)
ok("and says so", any("not a legal run" in n for n in _sp2["notes"]))
_sp3 = cp.scene_plan([{"frames": 192}, {"frames": 22, "continuity": "carry"}],
                     context=39)
ok("a pin cannot eat a whole scene", _sp3["scenes"][1]["pin"] < 22)
ok("off-grid scenes are named", any("audio grid" in n for n in
                                    cp.scene_plan([{"frames": 56}],
                                                  context=39)["notes"]))

if fails:
    print(f"\nFAIL — {len(fails)} check(s)")
    sys.exit(1)
print("\ncontinuity: the vocabulary holds")
