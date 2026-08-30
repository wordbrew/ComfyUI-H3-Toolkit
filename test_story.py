"""H3 Shot List and H3 Dialogue — the arithmetic that was being done on paper.

No torch, no ComfyUI. The one invariant worth guarding above all the others:
NO LINE IS EVER SCHEDULED BEFORE THE PIN CLEARS. The first `pin` frames of a
chunk are reproduced from the previous one under a denoise mask of 0, so speech
placed there is simply never said -- measured 2026-08-28, a line at 00:01.500
behind a 39-frame pin vanished and it took a finished render to notice.

    python3 test_story.py
"""

import importlib
import importlib.util
import pathlib
import sys
import types

# story.py imports from .chunkplan, so it only loads inside the package. Build
# the package with a stubbed torch -- the same approach test_seamcheck.py uses --
# rather than flattening the module's imports just to be testable.
_t = types.ModuleType("torch"); _t.__path__ = []
_nn = types.ModuleType("torch.nn"); _nn.__path__ = []
_fn = types.ModuleType("torch.nn.functional")
_t.nn = _nn; _nn.functional = _fn
for _n, _m in (("torch", _t), ("torch.nn", _nn), ("torch.nn.functional", _fn)):
    sys.modules.setdefault(_n, _m)
_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location(
    "h3story_pkg", _root / "__init__.py", submodule_search_locations=[str(_root)])
_pk = importlib.util.module_from_spec(_spec); sys.modules["h3story_pkg"] = _pk
try:
    _spec.loader.exec_module(_pk)
except Exception:
    pass                       # some nodes need ComfyUI; the arithmetic does not
plan = importlib.import_module("h3story_pkg.chunkplan").plan
_story = importlib.import_module("h3story_pkg.story")
FPS, H3Dialogue, H3Shotlist = _story.FPS, _story.H3Dialogue, _story.H3Shotlist
chunk_windows, parse_rows = _story.chunk_windows, _story.parse_rows
stamp, syllables = _story.stamp, _story.syllables

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


def plan_of(total, size, cuts=None, ctx=39):
    chunks, _ = plan(total, size, "scene" if cuts else "fixed", cuts=cuts,
                     context=ctx, grow_tail=True, generated_audio=True)
    return {"chunks": chunks}


# --- helpers ---------------------------------------------------------------- #
print("syllable estimate ranks lines by how long they take to say")
ok("a long word beats two short ones",
   syllables("unquestionably") > syllables("I know"))
check("silent e is not counted", syllables("time"), 1)
check("a vowel run is one syllable", syllables("queue"), 1)
ok("punctuation is ignored", syllables("Tell him eight o'clock.") >= 5)

print("timestamps are rounded to a tenth — the source is an estimate")
check("stamp rounds", stamp(3.3261), "00:03.300")
check("stamp crosses a minute", stamp(65.52), "01:05.500")
check("stamp at zero", stamp(0.0), "00:00.000")

print("rows split on the pipe and drop comments")
check("comments and blanks go",
      parse_rows("a | b\n\n# note\nc | d"), [["a", "b"], ["c", "d"]])

# --- the pin floor ---------------------------------------------------------- #
print("the speech window starts after the pin, never inside it")
p = plan_of(720, 243)                      # chained: pins of 39 after the first
w = chunk_windows(p["chunks"])
check("chunk 1 has no pin to clear", p["chunks"][0]["pin"], 0)
for i, (c, x) in enumerate(zip(p["chunks"], w), 1):
    if c["pin"]:
        ok(f"chunk {i} window starts past its {c['pin']}f pin",
           x["lo"] >= c["pin"] / FPS)
ok("and never past the end of the chunk", all(x["hi"] <= x["run_s"] for x in w))

# joined offsets: chunk k's local t=0 maps to where its kept frames begin, less
# the head the join throws away
kept, pos = [], 0
for c in p["chunks"]:
    kept.append(pos)
    pos += c["end"] - c["keep_from"]
for i, (c, x) in enumerate(zip(p["chunks"], w)):
    want = kept[i] / FPS - (c["keep_from"] - c["start"]) / FPS
    check(f"chunk {i + 1} joined offset", round(x["offset"], 4), round(want, 4))

# --- shot list -------------------------------------------------------------- #
print("shot list: legal runs, cuts at shot starts, one clause per chunk")
sl = H3Shotlist()
r = sl.go("10 | wide | they walk\n10 | medium | she turns\n10 | close | she stops",
          243, "39")["result"]
total, cutstr, beats, info = r
check("10s snaps to a legal run each time", total, 729)
check("cuts land on shot starts", cutstr, "243, 486")
check("one clause line per chunk, plus its comment",
      len([l for l in beats.splitlines() if not l.startswith("#")]), 3)
ok("the scaffold names the framing", "wide" in beats and "close" in beats)
ok("and says what it rounded", "not a legal run" in info)

# a shot longer than the chunk size splits, and each part needs its own clause
r2 = sl.go("30 | one long move | she walks the whole length", 243, "39")["result"]
chunks2, _ = plan(r2[0], 243, "scene", cuts=[], context=39, grow_tail=True,
                  generated_audio=True)
check("a long shot splits into several chunks",
      len([l for l in r2[2].splitlines() if not l.startswith("#")]),
      len(chunks2))

# --- dialogue --------------------------------------------------------------- #
print("dialogue: nothing lands inside a pin, breaks are honoured")
d = H3Dialogue()
res = d.go("S1 | says | He texted me at two in the morning.\n"
           "S2 | asks | And you answered?\n"
           "\n"
           "S1 | says | I let him wait until noon.\n"
           "S2 | says | That is mean.",
           plan_of(720, 243), speaker_map="S2=<Subject 3>")["result"]
beats, timeline, lint = res
check("one clause per chunk", len(beats.splitlines()), 4)
ok("the speaker map is applied", "<Subject 3> (S2)" in beats)
ok("the first anchor is capitalised", beats.splitlines()[0].startswith("At 00:"))
ok("later anchors are not", " at 00:" in beats.splitlines()[0])
ok("dialogue is marked up for the model", "<d>[English]" in beats)

# THE invariant: parse every emitted timestamp back and check it against the pin
import re
for i, (line, c) in enumerate(zip(beats.splitlines(), plan_of(720, 243)["chunks"]), 1):
    for mm, ss in re.findall(r"at (\d\d):(\d\d\.\d+)", line, re.I):
        t = int(mm) * 60 + float(ss)
        if t < c["pin"] / FPS:
            fails.append(f"chunk {i}: a line at {t:.3f}s is inside the "
                         f"{c['pin']}f pin ({c['pin'] / FPS:.3f}s)")
ok("NO emitted line falls inside its chunk's pin", not fails)

# a chunk with nothing in it is called out rather than left to repeat
res = d.go("S1 | says | Only one line.", plan_of(720, 243))["result"]
ok("an empty chunk is reported", "EMPTY" in res[0] or "nothing in it" in res[2])

# Silence across a seam is reported as DATA — it is the thing that reads as a
# fault when it lands mid-exchange, and the floor is the pin, so it cannot be
# removed, only placed. Needs lines in at least two chunks to have a gap at all.
spread = d.go("S1 | says | He texted me at two in the morning.\n"
              "S2 | asks | And you answered?\n"
              "\n"
              "S1 | says | I let him wait until noon.\n"
              "\n"
              "S2 | says | That is mean.\n"
              "\n"
              "S1 | says | Do not say a word.",
              plan_of(720, 243))["result"]
ok("seam silence is measured", "gap " in spread[2])
ok("and it names where", "between chunk 1 and chunk 2" in spread[2])

# SPREAD vs FRONT. Two short lines in a ten-second window front-loaded to 0.6s
# and 1.7s leave eight seconds of silence behind them; spreading puts the second
# one where it can be heard as a beat rather than a stammer.
import re as _re
def _times(beats_text):
    return [[int(a) * 60 + float(b)
             for a, b in _re.findall(r"at (\d\d):(\d\d\.\d+)", line, _re.I)]
            for line in beats_text.splitlines()]
two = ("S1 | says | Oh god, yes.\nS1 | says | Just like that.\n\n"
       "S1 | says | Do not stop.\nS1 | says | Harder.")
pl = plan_of(720, 243)
front = _times(d.go(two, pl, pacing="front")["result"][0])
spread = _times(d.go(two, pl, pacing="spread")["result"][0])
ok("front-loading puts the second line right behind the first",
   front[0][1] - front[0][0] < 1.5)
ok("spread pushes it out", spread[0][1] - spread[0][0] > front[0][1] - front[0][0])
ok("but never past the chunk's window",
   all(t <= chunk_windows(pl["chunks"])[i]["hi"] + 1e-6
       for i, ts in enumerate(spread) for t in ts))
ok("and never before the pin",
   all(t >= chunk_windows(pl["chunks"])[i]["lo"] - 1e-6
       for i, ts in enumerate(spread) for t in ts))
# max_gap is what stops spread becoming the same fault mirrored
wide = _times(d.go(two, pl, pacing="spread", max_gap=30.0)["result"][0])
tight = _times(d.go(two, pl, pacing="spread", max_gap=1.0)["result"][0])
ok("max_gap caps the silence spread inserts",
   (tight[0][1] - tight[0][0]) < (wide[0][1] - wide[0][0]))

# a line that cannot fit anywhere says so instead of being dropped
res = d.go("S1 | says | " + ("word " * 400), plan_of(243, 243))["result"]
ok("an overlong line is reported", "overflow" in res[2].lower())

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("story: all checks pass")
