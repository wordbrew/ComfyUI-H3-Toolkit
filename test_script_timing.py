"""timing() — which lines the chunking will eat, shown before a render.

THE FAILURE IT MAKES VISIBLE
  Every chunk after the first reproduces its opening `pin` frames from the
  previous one under a denoise mask of 0. Nothing new happens there, so that
  time is not sayable. Measured 2026-08-28, and invisible until you listen to a
  finished render.

  Lines are always placed at or after a chunk's floor, so none can literally
  start inside a pin. The damage shows up one step removed — as speech time that
  does not exist, and therefore as lines falling off the end of the take. Both
  forms are asserted here, because the warning is only useful if it names the
  real mechanism rather than a plausible-sounding one.

  timing() reuses chunkplan.plan and story.chunk_windows, the same two pieces
  H3Dialogue uses, so this shows what will happen rather than a second opinion.

    python3 test_script_timing.py
"""
import importlib.util
import pathlib
import sys
import types

_root = pathlib.Path(__file__).parent.resolve()
_pkg = types.ModuleType("h3tpk")
_pkg.__path__ = [str(_root)]
sys.modules["h3tpk"] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(f"h3tpk.{name}", _root / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"h3tpk.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_load("chunkplan")
_load("story")
hs = _load("h3script")

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


def take(*lines):
    body = "".join(f"  say @ada {l}\n" for l in lines)
    return hs.parse("@ada = a woman\nshot | a shot\n" + body)


def two_shots(placed_at):
    """Two shots, the second holding one line placed by hand at `placed_at`."""
    return hs.parse("@ada = a woman\nshot 141 | one\n  say @ada hi\n"
                    f"shot 243 | two\n  say @ada @{placed_at} asks Do you want to play?\n")


print("the first chunk has no pin; every one after it does")
t = hs.timing(take("hello"), total_frames=345, chunk_frames=141, context=39)
check("three chunks over 345 frames", len(t["chunks"]), 3)
check("chunk 0 is unpinned", t["chunks"][0]["pin_s"], 0.0)
ok("chunk 1 is pinned", t["chunks"][1]["pin_s"] > 0)
ok("a pinned chunk cannot be spoken in from t=0",
   t["chunks"][1]["speech_from"] >= t["chunks"][1]["pin_s"])

print("the pin is reported as time the take cannot use")
ok("pinned_seconds is the sum of the pins",
   abs(t["pinned_seconds"] - sum(c["pin_s"] for c in t["chunks"])) < 0.01)
ok("and it is stated as a problem",
   any("carried handles cost" in p for p in t["problems"]))
ok("sayable_seconds is less than the take",
   t["sayable_seconds"] < 345 / 24.0)

print("a line that fits is placed, and joined time is not local time")
t = hs.timing(take("I told you I would come back."),
              total_frames=345, chunk_frames=141, context=39)
first = t["lines"][0]
check("it lands in the first chunk", first["chunk"], 0)
check("nothing is wrong with it", first["problem"], None)
ok("it has a duration", first["seconds"] > 0)

print("more dialogue than the take can hold is reported, not silently dropped")
many = take(*["a reasonably long spoken line that takes a while to say"] * 12)
t = hs.timing(many, total_frames=345, chunk_frames=141, context=39)
overflow = [l for l in t["lines"] if l["problem"]]
ok("some lines overflow", len(overflow) > 0)
ok("every line is still accounted for", len(t["lines"]) == 12)
ok("the overflow says why", any("no room left" in (l["problem"] or "")
                                for l in overflow))

print("a shot with no declared length still spans the whole take")
t2 = hs.timing(many, total_frames=345 * 2, chunk_frames=141, context=39)
ok("fewer problems with more room",
   len([l for l in t2["lines"] if l["problem"]]) <
   len([l for l in t["lines"] if l["problem"]]))

print("no dialogue is not a problem")
t = hs.timing(hs.parse("@ada = a woman\nshot | a silent shot\n"),
              total_frames=141, chunk_frames=141, context=39)
check("no lines", t["lines"], [])
check("one chunk, so no pin and no complaint about one",
      any("carried handles cost" in p for p in t["problems"]), False)

print("the SERVER owns 17n+5 — a shot off the grid is snapped, and says so")
off = hs.parse("@a = x\nshot 150 | one\n  say @a hi\nshot 141 | two\n")
t_off = hs.timing(off, chunk_frames=141, context=39)
# 150 sits 9 above 141 and 8 below 158, so it goes UP — the nearest legal run,
# not the one below. Worth pinning: an earlier expectation here assumed "snap
# down" and was simply wrong about which was closer.
check("150 goes to the nearer run, which is 158", t_off["shots"][0]["frames"], 158)
check("and what was asked for is reported", t_off["shots"][0]["requested"], 150)
check("a legal length is left alone", t_off["shots"][1]["frames"], 141)
check("146 goes down to 141", hs.snap_shot(146), 141)
check("160 goes down to 158", hs.snap_shot(160), 158)
check("a legal run is untouched", hs.snap_shot(141), 141)
check("below the minimum clamps to 5", hs.snap_shot(1), 5)

print("times are given to a tenth, not a hundredth")
tt = hs.timing(take("I told you I would come back."), total_frames=345,
               chunk_frames=141, context=39)
ok("a line's start has at most one decimal",
   all(round(l["start"], 1) == l["start"] for l in tt["lines"] if l["start"] is not None))
ok("and so does its duration",
   all(round(l["seconds"], 1) == l["seconds"] for l in tt["lines"]))

print("shot lengths drive the cuts, and a cut opens an unpinned chunk")
doc = hs.parse("@ada = a woman\nshot 141 | one\n  say @ada hi\n"
               "shot 243 | two\n  say @ada there\n")
t = hs.timing(doc, chunk_frames=141, context=39)
check("the cut sits at the first shot's end", t["cuts"], [141])
check("total comes from the shots", t["total_frames"], 384)
check("the first chunk carries nothing", t["chunks"][0]["pin"], 0)
ok("a chunk is filed under the shot its KEPT region belongs to",
   t["chunks"][0]["shot"] == 0 and t["chunks"][1]["shot"] == 1)

print("a line placed by hand CAN land in a carried handle — the silent failure")
l = [x for x in hs.timing(two_shots("0"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
ok("placed at the very start of a pinned chunk is caught",
   "carried handle" in (l["problem"] or ""))
l = [x for x in hs.timing(two_shots("1.0"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
check("clear of the handle is fine", l["problem"], None)
l = [x for x in hs.timing(two_shots("9.9"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
ok("past the chunk's end is cut off, and says so",
   "cut off" in (l["problem"] or ""))

print("a FLOWED line is never accused of starting in a handle")
t = hs.timing(take("one", "two", "three"), total_frames=345,
              chunk_frames=141, context=39)
ok("no flowed line claims a handle problem",
   not any("carried handle" in (l["problem"] or "") for l in t["lines"]))

# these need the package loader above, because build_plan imports .chunkplan
print("the emitted plan is the one the board drew")
doc = hs.parse("@ada = a woman\nshot 141 | one\n  say @ada hi\n"
               "shot 243 | two\n  say @ada there\n")
chunks, _ = hs.build_plan(doc, chunk_frames=141, context=39)
t_ = hs.timing(doc, chunk_frames=141, context=39)
check("same number of chunks either way", len(chunks), len(t_["chunks"]))
check("and the same kept regions",
      [c["keep_from"] for c in chunks], [c["start"] for c in t_["chunks"]])

print("a length off the audio clock is called out, a legal one is not")
ok("141 is clean", not hs.audio_clock_notes(hs.parse(
    "@ada = a woman\nshot 141 | one\n")))
ok("142 is not a legal run", any("legal run" in n for n in hs.audio_clock_notes(
    hs.parse("@ada = a woman\nshot 142 | one\n"))))
ok("124 is legal but off the audio clock",
   any("audio clock" in n for n in hs.audio_clock_notes(
       hs.parse("@ada = a woman\nshot 124 | one\n"))))
print("the node runs end to end and emits a usable plan")
import json as _json
_res = hs.H3Script().go("@ada = character Ada\nshot 141 | one\n"
                        "  say @ada hi\nshot 243 | two\n  say @ada there\n")["result"]
check("document output is valid json", _json.loads(_res[8])["version"], 1)
check("cut_frames names the shot boundary", _res[10], "141")
check("total_frames is the take", _res[11], 384)
# THE SHAPE IS THE CONTRACT. H3_CHUNK_PLAN is a socket LABEL, not a checked
# type -- emitting a bare list passed every wiring check and died inside
# H3 Chunk Open with "'list' object has no attribute 'get'".
ok("the plan is the dict H3 Chunk Plan emits, not a bare list",
   isinstance(_res[12], dict))
for key in ("chunks", "info", "total_frames"):
    ok(f"it carries {key}", key in _res[12])
ok("and its chunks are chunk dicts",
   isinstance(_res[12]["chunks"], list) and "keep_from" in _res[12]["chunks"][0])
ok("a consumer's own access pattern works",
   (_res[12] or {}).get("chunks") is not None)
check("lora_schedule output exists", isinstance(_res[13], str), True)

print("per-shot loras become a schedule in slots, on the FINISHED clip")
ldoc = hs.parse("@ada = a woman\nshot 141 | one\n  lora h3/A.safetensors 0.8\n"
                "  say @ada hi\nshot 243 | two\n  lora h3/B.safetensors 0.4 -> 0.9\n"
                "  say @ada there\n")
lchunks, _ = hs.build_plan(ldoc, chunk_frames=141, context=39)
groups = hs.lora_schedule(ldoc, lchunks)
check("two files fit one node", len(groups), 1)
sched, slots = groups[0]
check("two files take two slots", slots,
      {"lora_1": "h3/A.safetensors", "lora_2": "h3/B.safetensors"})
ok("the schedule names SLOTS, never filenames",
   "lora_1" in sched and ".safetensors" not in sched)
ok("a ramp survives", "0.4-0.9" in sched)
check("one row per scheduled lora", len(sched.splitlines()), 2)
ok("the first shot starts the finished clip at 00:00", sched.startswith("00:00-"))

print("three slots per NODE is not a ceiling — more means another node")
four = hs.parse("@a = x\n" + "".join(
    f"shot 141 | s{i}\n  lora h3/L{i}.safetensors 1.0\n" for i in range(7)))
fchunks, _ = hs.build_plan(four, chunk_frames=141, context=39)
fg = hs.lora_schedule(four, fchunks)
check("seven loras become three nodes", len(fg), 3)
check("the first two are full", [len(g[1]) for g in fg[:2]], [3, 3])
check("the last holds the remainder", len(fg[2][1]), 1)
ok("every node numbers its own pickers from 1",
   all(set(g[1]) <= {"lora_1", "lora_2", "lora_3"} for g in fg))
ok("no lora is dropped",
   sum(len(g[1]) for g in fg) == 7)
ok("each node gets its own schedule rows", all(g[0].strip() for g in fg))

ok("the default script in the widget actually parses",
   hs.parse(hs.H3Script.INPUT_TYPES()["required"]["script"][1]["default"])
   is not None)


print()
print(f"{len(fails)} failure(s)" if fails else "script timing: all checks pass")
sys.exit(1 if fails else 0)
