"""H3 Chunk LoRA — the time arithmetic, which is the whole risk.

WHAT GOES WRONG IF THIS IS WRONG
  The schedule is written in FINISHED-CLIP time, and a chunk's own frames are
  not its output frames: the join drops everything before `keep_from`, so a
  chunk's local t=0 never appears. Getting that offset wrong puts a LoRA on the
  wrong chunk, and the render still completes — it just performs the wrong
  thing at the wrong moment, which is unfalsifiable by eye on a take you have
  not seen before.

  So the offset here is computed with `chunk_windows`, the same helper the
  dialogue node uses. These tests pin the two together: a cue and a spoken line
  at 00:12 have to land in the same chunk, or one of the two schedules is lying.

    python3 test_chunklora.py
"""
import importlib
import importlib.util
import pathlib
import sys
import types

_t = types.ModuleType("torch"); _t.__path__ = []
_nn = types.ModuleType("torch.nn"); _nn.__path__ = []
_fn = types.ModuleType("torch.nn.functional")
_t.nn = _nn; _nn.functional = _fn
for _n, _m in (("torch", _t), ("torch.nn", _nn), ("torch.nn.functional", _fn)):
    sys.modules.setdefault(_n, _m)

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location(
    "h3cl", _root / "__init__.py", submodule_search_locations=[str(_root)])
_pk = importlib.util.module_from_spec(_spec)
sys.modules["h3cl"] = _pk
try:
    _spec.loader.exec_module(_pk)
except Exception:
    pass

cl = importlib.import_module("h3cl.chunklora")
story = importlib.import_module("h3cl.story")
plan = importlib.import_module("h3cl.chunkplan").plan

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


def near(label, got, want, tol=1e-6):
    ok(f"{label} ({got!r} ~ {want!r})", abs(got - want) < tol)


# --- parsing ---------------------------------------------------------------- #
print("times read as MM:SS or bare seconds")
near("00:12", cl.parse_time("00:12"), 12.0)
near("01:02.5", cl.parse_time("01:02.5"), 62.5)
near("bare seconds", cl.parse_time("7.5"), 7.5)
check("nonsense is None", cl.parse_time("later"), None)

print("spans, including a bare start meaning 'from here on'")
check("00:00-00:12", cl.parse_span("00:00-00:12"), (0.0, 12.0))
check("dotted form", cl.parse_span("3..9"), (3.0, 9.0))
check("' to '", cl.parse_span("3 to 9"), (3.0, 9.0))
ok("a bare time runs to the end", cl.parse_span("00:30")[1] == float("inf"))
check("unreadable is None", cl.parse_span("whenever"), None)

print("strengths, flat or ramped")
check("flat", cl.parse_strength("0.8"), (0.8, 0.8))
check("ramp", cl.parse_strength("0.4-0.9"), (0.4, 0.9))
check("garbage falls back to 1.0", cl.parse_strength(""), (1.0, 1.0))

# --- the offset, which is the part that bites ------------------------------- #
print("a chunk's span is its KEPT frames on the finished clip, not its own")
chunks, _ = plan(720, 243, "fixed", context=39, grow_tail=True,
                 generated_audio=True)
spans = cl.chunk_spans(chunks)
near("the take starts at zero", spans[0][0], 0.0)
ok("spans are contiguous, no gap and no overlap",
   all(abs(spans[k][1] - spans[k + 1][0]) < 1e-6 for k in range(len(spans) - 1)))
total = sum(int(c["end"]) - int(c["keep_from"]) for c in chunks) / story.FPS
near("and they cover the whole clip", spans[-1][1], total)
ok("a pinned chunk's span starts AFTER the previous one ends, not at its "
   "own start frame",
   spans[1][0] > int(chunks[1]["start"]) / story.FPS)

print("a cue and a spoken line at the same time land in the same chunk")
wins = story.chunk_windows(chunks)
for t in (0.5, 5.0, 12.0, 18.0, 25.0):
    by_span = [k for k, (a, b) in enumerate(spans) if a <= t < b]
    # the dialogue node places a line at joined time `offset + local`
    by_dialogue = [k for k, w in enumerate(wins)
                   if w["offset"] + w["lo"] <= t <= w["offset"] + w["hi"]]
    ok(f"t={t}s agrees" if by_span else f"t={t}s is inside the clip",
       bool(by_span) and (not by_dialogue or by_span[0] in by_dialogue))

# --- resolution ------------------------------------------------------------- #
print("rows apply to every chunk they overlap")
rows = [("a.safetensors", (0.0, 10.0), (0.8, 0.8)),
        ("b.safetensors", (10.0, 40.0), (0.4, 0.9))]
first = cl.resolve(rows, spans[0])
ok("chunk 1 gets a", any(n == "a.safetensors" for n, _ in first))
check("at its flat strength", dict(first)["a.safetensors"], 0.8)
last = cl.resolve(rows, spans[-1])
ok("the last chunk gets b", any(n == "b.safetensors" for n, _ in last))

print("a ramp rises across the take and is evaluated at each chunk's midpoint")
vals = [dict(cl.resolve([rows[1]], s)).get("b.safetensors") for s in spans]
got = [v for v in vals if v is not None]
ok("more than one chunk sees it", len(got) > 1)
ok("and it increases", all(b >= a for a, b in zip(got, got[1:])))
ok("staying inside the endpoints", all(0.4 - 1e-6 <= v <= 0.9 + 1e-6
                                       for v in got))

print("a row nobody overlaps applies to nobody")
check("far future", cl.resolve([("z", (9000.0, 9100.0), (1.0, 1.0))],
                               spans[0]), [])

print("a bare start time runs to the end of the take")
tail = [("c", cl.parse_span("00:20"), (1.0, 1.0))]
ok("the last chunk is covered", cl.resolve(tail, spans[-1]) != [])
ok("the first is not", cl.resolve(tail, spans[0]) == [])

# --- the schema contract ---------------------------------------------------- #
print("wiring contract")
t = cl.H3ChunkLora.INPUT_TYPES()
check("required order", list(t["required"]), ["model", "chunk_index", "schedule"])
ok("chunk_index is forceInput — it is what puts the node in the chunk body",
   t["required"]["chunk_index"][1].get("forceInput") is True)
check("returns", cl.H3ChunkLora.RETURN_NAMES, ("model", "info"))

print("no chunk_plan is an error, not a silent pass-through")
try:
    cl.H3ChunkLora().go(object(), 0, "00:00-00:10 | a.safetensors | 0.5")
    check("raises", "no error", "ValueError")
except ValueError as e:
    ok("and it says why", "FINISHED-CLIP time" in str(e) or "chunk_plan" in str(e))

print("an empty schedule passes the model through untouched")
sentinel = object()
res = cl.H3ChunkLora().go(sentinel, 0, "# only a comment\n")
ok("same object back", (res["result"] if isinstance(res, dict) else res)[0]
   is sentinel)

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("chunk lora: all checks pass")
