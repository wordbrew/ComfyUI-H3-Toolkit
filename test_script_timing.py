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

print()
print(f"{len(fails)} failure(s)" if fails else "script timing: all checks pass")
sys.exit(1 if fails else 0)
