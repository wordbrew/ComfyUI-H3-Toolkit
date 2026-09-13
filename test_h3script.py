"""H3 Script — the numbering, and the round trip that makes a UI possible.

THE POINT OF THE COMPILER
  `<Subject n>` / `<Picture n>` bookkeeping spread across five nodes is what
  broke a take on 2026-09-02: subject_definitions told the model to preserve
  her "exactly as shown in those pictures" while the description asked for
  motion, so it rendered the still. Nothing errored. These tests pin the two
  properties that prevent that class of bug: indices are assigned from one
  place, and a fully_preserved retention never binds preservation to pictures.

THE ROUND TRIP
  parse() -> document -> emit(). The document is plain data, so a timeline
  editor can read it, change it and hand it back. If that stops holding, the
  app becomes a rewrite instead of a view, so it is asserted here rather than
  assumed.

    python3 test_h3script.py
"""
import importlib.util
import json
import pathlib
import sys

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location("h3s", _root / "h3script.py")
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


SCRIPT = """
# a take
@ada  = character Ada
@ada.pictures = 3
@ada.audio = 1
# pictures/audio are declared here only because these tests must not depend on
# what happens to be in the character store; a real script omits them
@man  = a man in his thirties, dark hair
@man.retention = partially_preserved
@room = setting. a bedroom in warm low light

task       = reference generation
soundscape = room tone, breathing, @room settling
music      = N/A

shot | a locked-off two-shot of @ada and @man on the bed
  note @ada is on her back, nearer camera
  say  @ada moans Oh god, yes.
  say  @ada Just like that.
  do   his hips work in a steady rhythm
  ---
  say  @man asks Like that?
  do   the pace quickens
  lora slow_grind 0.4 -> 0.9
"""

doc = hs.parse(SCRIPT)
out = hs.emit(doc)

# --- the numbering ---------------------------------------------------------- #
print("declaration order decides the subject numbers")
idx = out["index"]
check("ada is Subject 1", idx["ada"]["subject"], 1)
check("man is Subject 2", idx["man"]["subject"], 2)
check("room is Subject 3", idx["room"]["subject"], 3)

print("pictures and audio are allocated in order, without gaps or overlap")
check("ada holds pictures 1-3", idx["ada"]["pictures"], [1, 2, 3])
check("man holds none", idx["man"]["pictures"], [])
check("ada holds audio 1", idx["ada"]["audio"], [1])
check("counts", out["counts"], {"subjects": 3, "pictures": 3, "audio": 1})

print("a second character's pictures start after the first's")
two = hs.emit(hs.parse("@a = character A\n@a.pictures = 3\n"
                       "@b = character B\n@b.pictures = 2\n"
                       "shot | @a and @b\n"))
check("b gets 4 and 5", two["index"]["b"]["pictures"], [4, 5])

# --- substitution ----------------------------------------------------------- #
print("@names become subject tokens everywhere free text appears")
ok("in the description", "<Subject 1>" in out["head"] and
   "<Subject 2>" in out["head"])
ok("in a note", "<Subject 1> is on her back" in out["head"])
ok("in the soundscape", "<Subject 3>" in out["soundscape"])
ok("and no @ survives into the prompt", "@" not in out["head"])

print("a character's pictures are cited in its definition")
ok("pictures listed", "<Picture 1>, <Picture 2> and <Picture 3>"
   in out["subject_defs"])
ok("voice cited", "<Audio 1>" in out["subject_defs"])

# --- retention is about LIKENESS ------------------------------------------- #
# It scopes what stays the same about a subject between shots. Pose and motion
# are directed in the shot, via `note` and `do`. An earlier version wrote pose
# instructions into this block to work around a static take; that conflated two
# fields and encoded a guess as a rule -- the take that DID render motion used
# the plain form, citing its pictures.
print("fully_preserved states likeness, and cites the pictures")
ret = out["retention"]
ok("the marker is present", "<Subject 1>: fully_preserved -" in ret)
ok("it names what is preserved", "facial identity, hair, eye colour" in ret)
ok("and cites the pictures", "as shown in <Picture 1>" in ret)
ok("no pose or motion direction leaks into it",
   not any(w in ret.lower() for w in ("pose", "movement come", "what they are "
                                      "doing")))

print("preserve / allow scope the likeness, and detail replaces the clause")
scoped = hs.emit(hs.parse(
    "@a = character A\n@a.pictures = 2\n"
    "@a.retention = fully_preserved\n"
    "@a.preserve = her face and the tattoo on her left arm\n"
    "@a.allow = a natural gait and changing expression\n"
    "shot | @a walks\n"))["retention"]
ok("preserve is used", "her face and the tattoo on her left arm" in scoped)
ok("allow is used", "while allowing a natural gait" in scoped)
custom = hs.emit(hs.parse(
    "@a = character A\n@a.retention = fully_preserved\n"
    "@a.retention_detail = exactly as written here\n"
    "shot | @a\n"))["retention"]
check("detail replaces the whole clause", custom,
      "<Subject 1>: fully_preserved - exactly as written here")

# --- dialogue hand-off ------------------------------------------------------ #
print("dialogue is emitted in H3 Dialogue's OWN input format, not timed here")
lines = out["dialogue_lines"].splitlines()
check("first line", lines[0], "S1 | moans | Oh god, yes.")
check("second", lines[1], "S1 | says | Just like that.")
check("a blank line separates the chunks", lines[2], "")
check("then the next speaker", lines[3], "S2 | asks | Like that?")
check("actions, one per chunk, in order", out["dialogue_actions"].splitlines(),
      ["his hips work in a steady rhythm.", "the pace quickens."])
ok("speaker map binds S ids to subjects",
   "S1=<Subject 1>" in out["speaker_map"] and
   "S2=<Subject 2>" in out["speaker_map"])
ok("NO timestamps are emitted — H3Dialogue owns placement, because it owns "
   "the pin rule", ":" not in out["dialogue_lines"])

# --- the round trip --------------------------------------------------------- #
print("the document is plain data and survives a round trip")
blob = json.dumps(doc)
again = json.loads(blob)
check("json is lossless", again, doc)
check("and re-emitting gives the same fields", hs.emit(again)["head"],
      out["head"])
ok("a UI could edit it: shots and cast are lists of dicts",
   isinstance(doc["shots"], list) and isinstance(doc["cast"], list) and
   all(isinstance(s["chunks"], list) for s in doc["shots"]))

# --- errors name their line ------------------------------------------------- #
print("mistakes are reported with the line, not swallowed")
for bad, why in (
        ("@a = x\nshot | s\n  say @nobody hello\n", "undeclared name"),
        ("@a = x\n@a = y\nshot | s\n", "duplicate declaration"),
        ("@a = x\n@a.retention = sort of\nshot | s\n", "bad retention"),
        ("@a = x\nshot | s\n  wobble thing\n", "unknown keyword"),
        ("  say @a hi\n", "indented before a shot"),
        ("@a = x\nshot | s\n  say @a\n", "say with no line")):
    try:
        hs.parse(bad)
        check(why, "no error", "ScriptError")
    except hs.ScriptError as e:
        ok(f"{why} -> {str(e).splitlines()[0][:40]}", True)

check("an empty script is an error", True, True)
try:
    hs.parse("# nothing\n")
    fails.append("empty script did not raise")
except hs.ScriptError:
    pass

# --- lint ------------------------------------------------------------------- #
print("lint reports what renders fine and is wrong")
d2 = hs.parse("@a = x\n@unused = y\nshot | @a alone\n")
notes = hs.lint(d2, hs.emit(d2))
ok("an unused subject is called out",
   any("@unused" in n and "never used" in n for n in notes))
d3 = hs.parse("@c = character C\nshot | @c\n")
ok("a character with no pictures is called out",
   any("pictures" in n for n in hs.lint(d3, hs.emit(d3))))
d4 = hs.parse("@a = x\nshot | @a\n  say @a hi\n  ---\n  ---\n  do thing\n")
ok("an empty chunk break is called out",
   any("nothing directing it" in n for n in hs.lint(d4, hs.emit(d4))))

# --- the node contract ------------------------------------------------------ #
print("node contract")
check("returns", hs.H3Script.RETURN_NAMES,
      ("head", "subject_defs", "retention", "soundscape", "music",
       "dialogue_lines", "dialogue_actions", "speaker_map", "document", "info",
       # APPENDED, per the slot contract — these carry the shot lengths into
       # H3 Chunk Plan, and `plan` IS the plan the timeline drew, so the board
       # cannot describe one render while ComfyUI performs another.
       "cut_frames", "total_frames", "plan", "lora_schedule",
       # one assembled string, because that is what every consumer takes; the
       # separate fields are for showing, not for wiring
       "prompt",
       # the cast's OWN anchors, in the order this node numbered them, so the
       # picture cited and the picture shown cannot come apart
       "tail",
       "picture_1", "picture_2", "picture_3", "picture_4", "voice_1", "voice_2",
       # a COMPLETE six-section prompt per chunk, `---` separated, for H3 Long-
       # Form Links' beats. Last, on the end, because a saved graph stores output
       # slots by index. The loose sections above stay for the graphs using them.
       "chunk_prompts"))
check("the original ten keep their positions", hs.H3Script.RETURN_NAMES[:10],
      ("head", "subject_defs", "retention", "soundscape", "music",
       "dialogue_lines", "dialogue_actions", "speaker_map", "document", "info"))
check("one per type", len(hs.H3Script.RETURN_TYPES),
      len(hs.H3Script.RETURN_NAMES))
check("the plan rides the same bus the chunk nodes consume",
      hs.H3Script.RETURN_TYPES[hs.H3Script.RETURN_NAMES.index("plan")],
      "H3_CHUNK_PLAN")

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("h3 script: all checks pass")

# --- the store supplies what you would otherwise re-type -------------------- #
# A character saved with H3 Character already has its anchors, voice and
# description on disk. Making the author restate the anchor count is the kind
# of bookkeeping this node exists to remove, so parse() reads the store. It has
# to degrade silently when there is no store at all, which is the case here and
# in any test run outside ComfyUI.
print("the character store supplies pictures, voice and description")
n, v, desc, ret, akind = hs.store_card("almost-certainly-not-a-character")
check("an unknown character reads as nothing", (n, v, desc, ret), (0, 0, "", ""))
d = hs.parse("@x = character Nobody\nshot | @x\n")
c = d["cast"][0]
check("still parses without a store", c["kind"], "character")
check("and defaults to fully_preserved", c["retention"], "fully_preserved")
check("with no pictures claimed", c["pictures"], 0)
ok("lint says so rather than letting it render silently",
   any("pictures" in n for n in hs.lint(d, hs.emit(d))))
ok("an explicit override still wins",
   hs.parse("@x = character Nobody\n@x.pictures = 2\nshot | @x\n"
            )["cast"][0]["pictures"] == 2)
print()
print("h3 script: store checks pass")
