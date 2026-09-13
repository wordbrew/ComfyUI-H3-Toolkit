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
    """Two shots; the second's line pinned to `placed_at` seconds of the FINISHED
    clip. Joined time, not shot-relative -- the same number the ruler shows."""
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
# chunk 2 begins at 5.875s joined and holds 1.625s back — a line pinned there is
# inside the handle
l = [x for x in hs.timing(two_shots("5.9"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
ok("pinned into a carried handle is caught",
   "carried handle" in (l["problem"] or ""))
# chunk 2 is speakable from 5.95s to 9.53s joined; a ~1.2s line pinned at 9.0
# would run to 10.16 and be cut off, so 7.0 is the one that actually fits
l = [x for x in hs.timing(two_shots("7.0"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
check("clear of the handle, and inside the ceiling, is fine", l["problem"], None)
check("and it is heard exactly where it was pinned", l["start"], 7.0)
l = [x for x in hs.timing(two_shots("10.0"), chunk_frames=141, context=39)["lines"]
     if x["placed"]][0]
ok("too near a chunk's end is cut off, and says so",
   "cut off" in (l["problem"] or ""))

print("the board and H3 Dialogue agree, because they share one packer")
import importlib
_st = sys.modules["h3tpk.story"]
_doc = hs.parse("@a = x\nshot 243 | one\n"
                "  say @a the first line of a short exchange\n"
                "  say @a and the second one answering it\n")
_plan = hs.plan_payload(_doc, chunk_frames=141, context=39)
_out = hs.emit(_doc)
_res = _st.H3Dialogue().go(lines=_out["dialogue_lines"], chunk_plan=_plan,
                           actions=_out["dialogue_actions"],
                           speaker_map=_out["speaker_map"])
_beats, _timeline, _lint = _res["result"] if isinstance(_res, dict) else _res
_board = [l["start"] for l in hs.timing(_doc, chunk_frames=141, context=39)["lines"]]
import re as _re
_heard = [round(float(x), 1) for x in
          _re.findall(r"^\s*([0-9.]+)-", _timeline, _re.M)]
check("the times the board shows are the times that get rendered",
      _board, _heard)

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

print("per-shot loras become a schedule of FILES, on the FINISHED clip")
ldoc = hs.parse("@ada = a woman\nshot 141 | one\n  lora h3/A.safetensors 0.8\n"
                "  say @ada hi\nshot 243 | two\n  lora h3/B.safetensors 0.4 -> 0.9\n"
                "  say @ada there\n")
lchunks, _ = hs.build_plan(ldoc, chunk_frames=141, context=39)
sched = hs.lora_schedule(ldoc, lchunks)
ok("it is one schedule, not a list of node-sized groups", isinstance(sched, str))
# NAMING THE FILE IS WHAT REMOVES THE HAND-WORK. H3 Chunk Lora loads a row whose
# name is not a picker slot by filename, so nothing has to be typed into a picker
# and there is no three-per-node ceiling to chain around. Both were my invention.
ok("the schedule names FILES, never picker slots",
   "h3/A.safetensors" in sched and "h3/B.safetensors" in sched
   and "lora_1" not in sched)
ok("a ramp survives", "0.4-0.9" in sched)
check("one row per scheduled lora", len(sched.splitlines()), 2)
ok("the first shot starts the finished clip at 00:00", sched.startswith("00:00-"))

print("and seven loras are still ONE schedule, because slots are not involved")
four = hs.parse("@a = x\n" + "".join(
    f"shot 141 | s{i}\n  lora h3/L{i}.safetensors 1.0\n" for i in range(7)))
fchunks, _ = hs.build_plan(four, chunk_frames=141, context=39)
fs = hs.lora_schedule(four, fchunks)
check("seven rows, one per lora", len(fs.splitlines()), 7)
ok("no chaining is implied anywhere in it", "lora_" not in fs)
ok("no lora is dropped",
   all(f"h3/L{i}.safetensors" in fs for i in range(7)))

ok("the default script in the widget actually parses",
   hs.parse(hs.H3Script.INPUT_TYPES()["required"]["script"][1]["default"])
   is not None)



# --- the six-section format, per chunk ------------------------------------- #
#
# WHAT THIS DEFENDS. On 2026-09-12 the chunked path was measured against
# docs/prompting-ref2va.md and diverged in five ways at once: the style
# paragraph never arrived, every chunk was handed the WHOLE take's shot list, the
# shot clock and the dialogue clock were different clocks in one paragraph, the
# continuity clause sat inside detailed_description, and summary carried "link 1
# of 2" bookkeeping. All five were invisible in a render.

print("a chunk's prompt describes THAT chunk, in the guide's six sections")
FULL = """@ada = a woman in her thirties
@ben = a man with grey hair
@ada.wears = a charcoal wool coat
@ada.voice = a low measured voice
@ada.pronoun = she
@room = setting. a concrete stairwell landing
style = Photorealistic live-action, 16:9, handheld on 35mm with visible grain
camera = handheld with small continuous drift, never a deliberate move
lips = auto
negatives = no other people, no readable text, no music
soundscape = close room tone in hard concrete with a long reflective tail

shot 141 | a two-shot of @ada facing @ben on the landing
  say @ada | says | You said tomorrow.
  say @ada | says at once | To when?
shot 141 | a close-up of @ben
  say @ben | says | Understood.
"""
fdoc = hs.parse(FULL)
fchunks2, _ = hs.build_plan(fdoc, chunk_frames=141, context=39)
parts = hs.chunk_prompts(fdoc, fchunks2).split(hs.CLAUSE_SEP)
check("one prompt per chunk", len(parts), len(fchunks2))
for i, p in enumerate(parts):
    for sec in ("subject_definitions:", "summary:", "retention_analysis:",
                "detailed_description:", "overall_soundscape:",
                "non_diegetic_music:"):
        ok(f"chunk {i} has {sec}", sec in p)
    ok(f"chunk {i} carries the look", "Photorealistic live-action" in p)
    ok(f"chunk {i} carries the lip discipline", "lips move only" in p)
    ok(f"chunk {i} says nothing about links", "link " not in p.lower())
    ok(f"chunk {i} numbers its first shot [Shot 1]", "[Shot 1]" in p)

print("the clock is the chunk's own, and only its own shots appear")
first, last = parts[0], parts[-1]
ok("the opening chunk's shot is untimed", "[Shot 1] A two-shot" in first)
ok("it does not mention the later shot", "close-up of <Subject 2>" not in first)
ok("the last chunk holds the later shot instead",
   "close-up of <Subject 2>" in last and "A two-shot" not in last)
ok("and no timestamp from the take's clock leaks into a chunk that cannot reach it",
   "00:05." not in first)

print("dialogue sits inside its shot, in the guide's verbatim form")
ok("subject, then (Sx), then the verb, then the tag",
   "<Subject 1>, with a low measured voice (S1), says: <d>[English] You said "
   "tomorrow.</d>" in first)
ok("a second line from the same speaker uses the pronoun",
   "She says at once: <d>[English] To when?</d>" in first)
ok("the id never trails the tag", "</d> (S" not in first)

print("summary names this window, not the take")
ok("the last chunk credits only the speaker who speaks in it",
   "<Subject 2> speaks 1 line" in last and "<Subject 1> and" not in
   last.split("detailed_description:")[0].split("summary:")[1])
ok("the continuity clause is in summary, not in the description",
   "unbroken" in last.split("detailed_description:")[0])

print("wardrobe is stated, because clothing follows the prompt not the anchors")
ok("the definition carries it",
   "<Subject 1> wears a charcoal wool coat in the target video." in first)

print("lint names each missing half of the look separately")
thin = hs.parse("@a = a woman\nshot 141 | a close-up of @a\n  say @a says hi\n")
notes = " | ".join(hs.lint(thin, hs.emit(thin)))
for want in ("`style =`", "`camera =`", "`negatives =`", "`.wears`"):
    ok(f"it asks for {want}", want in notes)
ok("and it reports the word count against the guide's range",
   "350-500" in notes)

print()
print(f"{len(fails)} failure(s)" if fails else "script timing: all checks pass")
sys.exit(1 if fails else 0)
