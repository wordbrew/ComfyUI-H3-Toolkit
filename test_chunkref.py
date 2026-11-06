#!/usr/bin/env python3
"""Per-chunk references, and the retention clause that makes depth work.

CJ's method, in his words: "I use depth based videos with specific prompting in
the retention piece of the prompt to drive motion and composition for clips ...
one prompts for using motion and composition." The depth pass alone does
nothing — without the clause the model reads a depth map as a thing to render.
So the clause is tested as carefully as the gating.
"""
import sys
import types

sys.path.insert(0, ".")
import audit_nodes as A                                        # noqa: E402

_classes, _fail = A.load_pack()
assert not _fail, _fail
cp = sys.modules["_h3audit.chunkplan"]
cr = sys.modules["_h3audit.chunkref"]
N = _classes["H3ChunkRef"]

fails = []


def ok(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


def check(label, got, want):
    ok(f"{label}: {got!r}", got == want)


_ch, _ = cp.plan(cp.total_for_count(192, 4, 39), 192, "fixed",
                 context=39, grow_tail=True)
PLAN = {"chunks": _ch}


def run(**kw):
    kw.setdefault("chunk_plan", PLAN)
    kw.setdefault("image", "IMG")
    return N().go(**kw)["result"]


print("a reference is held to its chunks")
check("live on the chunk it names",
      run(role="composition", chunks="1", chunk_index=0)[1], True)
check("and emits the image there",
      run(role="composition", chunks="1", chunk_index=0)[0], "IMG")
check("not live elsewhere",
      run(role="composition", chunks="1", chunk_index=1)[1], False)
check("and emits NOTHING there, so the slot stays empty",
      run(role="composition", chunks="1", chunk_index=1)[0], None)

print("\nranges, lists and `last` mean what they do in the LoRA schedule")
for spec, live in (("2-4", (False, True, True, True)),
                   ("1,3", (True, False, True, False)),
                   ("3+", (False, False, True, True)),
                   ("last", (False, False, False, True))):
    got = tuple(run(role="motion", chunks=spec, chunk_index=i)[1]
                for i in range(4))
    check(f"  {spec}", got, live)

print("\n`all` is the default and the old behaviour")
for spec in ("all", "", "   ", "every"):
    ok(f"{spec!r} is live everywhere",
       all(run(role="identity", chunks=spec, chunk_index=i)[1]
           for i in range(4)))

print("\nTHE CLAUSE IS THE POINT — depth pixels alone do nothing")
_c = run(role="composition", chunks="all", chunk_index=0, ordinal=5)[2]
ok("a composition reference says to follow the composition",
   "follow only the composition" in _c)
ok("and to ignore the subject", "ignore the subject" in _c)
ok("and names its own ordinal as a Picture", "<Picture 5>" in _c)
_m = run(role="motion", chunks="all", chunk_index=0, ordinal=2)[2]
ok("a motion reference asks for composition AND motion",
   "composition and motion" in _m)
ok("and names itself as a Video", "<Video 2>" in _m)
check("identity carries no clause, because it is cited in the subject "
      "definition instead",
      run(role="identity", chunks="all", chunk_index=0)[2], "")
ok("a reference that is not live says nothing either",
   run(role="motion", chunks="2", chunk_index=0)[2] == "")

print("\nthe ordinal follows the reference, so two do not both claim <Picture 1>")
ok("each names its own",
   run(role="composition", chunks="all", chunk_index=0, ordinal=1)[2]
   != run(role="composition", chunks="all", chunk_index=0, ordinal=3)[2])

print("\nit refuses to quietly do the wrong thing")
_r = run(role="composition", chunks="9", chunk_index=0)
ok("an out-of-range chunk is reported", "9" in _r[3])
ok("and the take still renders", "still renders" in _r[3])
_t = run(role="composition", chunks="0:04", chunk_index=0)
ok("a TIME is refused, since a reference cannot half-cover a chunk",
   "reads as a TIME" in _t[3])
_n = N().go(role="motion", chunks="all", chunk_index=0, chunk_plan=PLAN)["result"]
ok("live with nothing wired says so", "nothing is wired" in _n[3])
ok("and a depth role reminds you to pass it through depth first",
   "DEPTH pass upstream" in _r[3])


print("\nseveral references chain into one retention_1")
# H3LongFormLinks has ONE retention input and an input takes ONE wire, so the
# nodes chain: each appends its clause and passes the whole thing on.
_a = run(role="composition", chunks="all", chunk_index=0, ordinal=1)[2]
_b = run(role="motion", chunks="all", chunk_index=0, ordinal=1,
         retention_in=_a)[2]
check("two clauses, one per line", len(_b.splitlines()), 2)
ok("the earlier one is kept", "follow only the composition, ignore" in _b)
ok("and the later one added", "composition and motion" in _b)
_c = run(role="identity", chunks="all", chunk_index=0, retention_in=_b)[2]
check("a clause-less role passes the chain through unchanged", _c, _b)
_d = run(role="motion", chunks="2", chunk_index=0, retention_in=_a)[2]
check("and so does a reference that is not live here", _d, _a)

print("\nspanning a cut is WARNED, never refused")
# CJ: "yes, a reference should be able to span a cut. leave it to the user, i
# would say you warn them, but don't prevent."
_cut, _ = cp.plan(cp.total_for_count(192, 4, 39), 192, "fixed", context=39,
                  grow_tail=True, continuity=["carry", "cut", "carry"])
_cp = {"chunks": _cut}
_r = N().go(role="composition", chunks="1-4", chunk_index=0, image="I",
            ordinal=1, chunk_plan=_cp)["result"]
ok("it says which chunk the cut is at", "spans a cut at chunk 3" in _r[3])
check("and still emits the image", _r[0], "I")
check("and still emits the clause", bool(_r[2]), True)
_q = N().go(role="composition", chunks="1-2", chunk_index=0, image="I",
            ordinal=1, chunk_plan=_cp)["result"]
ok("a reference inside one shot is not warned", "spans a cut" not in _q[3])
ok("and a MOTION reference is not warned, since movement across a cut is "
   "a normal thing to ask for",
   "spans a cut" not in N().go(role="motion", chunks="1-4", chunk_index=0,
                               image="V", chunk_plan=_cp)["result"][3])

print("\nA CONTINUITY CUT IS A CUT, in both planners")
# plan() computed `cut` from the shot-part index alone, so a cut named through
# `continuity` carried the label and none of the behaviour -- seed_mask stayed
# True and the mask tracker seeded ACROSS the cut, which chunkrun's own
# docstring forbids. scene_plan had it right and the two disagreed.
check("plan() marks the cut chunk", [c["cut"] for c in _cut],
      [False, False, True, False])
check("and does not seed the mask across it",
      [c["seed_mask"] for c in _cut], [False, True, False, True])
_sp = cp.scene_plan([{"frames": 192},
                     {"frames": 192, "continuity": "carry"},
                     {"frames": 192, "continuity": "cut"},
                     {"frames": 192, "continuity": "carry"}],
                    context=39)["scenes"]
check("and scene_plan says the same thing",
      [(c["cut"], c["seed_mask"]) for c in _sp],
      [(c["cut"], c["seed_mask"]) for c in _cut])


print("\nFRAME 0 BELONGS TO A CHUNK THAT OPENS A SHOT")
# CJ: "obviously a frame 0 should only be applied to the very first chunk
# yeah?" Chunk 1 always opens one -- and so does any chunk after a CUT, whose
# opening frames are equally fresh. A CONTINUING chunk's opening frames ARE
# the pin, held by the mask, so an image anchored there competes with the hold
# that makes long-form work at all. That is a conflict, not a preference, so
# it refuses rather than warns.
_cut, _ = cp.plan(cp.total_for_count(192, 4, 39), 192, "fixed", context=39,
                  grow_tail=True, continuity=["carry", "cut", "carry"])
_cp = {"chunks": _cut}


def f0(i, chunks="all"):
    return N().go(role="frame 0", chunks=chunks, chunk_index=i, image="IMG",
                  ordinal=1, chunk_plan=_cp)["result"]


ok("chunk 1 anchors — the take opens there", f0(0)[0] == "IMG")
ok("chunk 3 anchors — it is the cut, so its opening is fresh",
   f0(2)[0] == "IMG")
ok("chunk 2 refuses — it continues, its opening is the pin", f0(1)[0] is None)
ok("chunk 4 refuses for the same reason", f0(3)[0] is None)

_r = f0(1)
ok("and it says WHY, naming the pin", "the pin" in _r[3])
ok("and which chunks WOULD take it", "opens a shot: 1, 3" in _r[3])

print("the default scope is the openers, not `all`")
check("so it lands where it is legal", cr.default_scope("frame 0", _cut), "1,3")
check("while a composition reference still defaults to everywhere",
      cr.default_scope("composition", _cut), "all")
ok("and with no plan at all it does not guess a cut",
   cr.default_scope("frame 0", []) == "all")

print("no other role is restricted")
for _role in ("composition", "composition and motion", "identity"):
    ok(f"{_role} is live on a continuing chunk",
       N().go(role=_role, chunks="all", chunk_index=1, image="I", ordinal=1,
              chunk_plan=_cp)["result"][0] == "I")


print("\nevery role that emits a clause names one of the model's markers")
# A marker the model does not know is just a word in the retention block.
# prompt_lint owns the vocabulary; the roles have to stay inside it.
import importlib.util as _iu
_ps = _iu.spec_from_file_location("_pl2", "prompt_lint.py")
_pl2 = _iu.module_from_spec(_ps); _ps.loader.exec_module(_pl2)
for _r in cr.ROLE_ORDER:
    _c = cr.retention_clause(_r, 1)
    if not _c:
        continue
    _marker = _c.split(":", 1)[1].split(" - ")[0].strip()
    ok(f"{_r} -> {_marker}", _marker in _pl2.VISUAL_MARKERS)

print("and frame 0 says it CONTINUES, not just that it matches")
# the recorded failure is the reference sitting as the first five frames and
# then the real shot starting -- a still followed by a cut
_f0 = cr.retention_clause("frame 0", 1)
ok("it asks for an opening, not a held frame", "continues from it" in _f0)
ok("and fully_preserved, because the image is the thing itself",
   "fully_preserved" in _f0)

print("\nLAST FRAME is frame 0's mirror, and the asymmetry is deliberate")
# frame 0 REFUSES on a continuing chunk because the pin owns those rows.
# Nothing is held at a chunk's TAIL, so the same anchor at the end is a
# direction -- "arrive at this pose" -- and only warrants a warning.
_lf = cr.retention_clause("last frame", 3)
ok("it names the final frame", "final frame of the video" in _lf)
ok("and says the shot ARRIVES there, not that a still is appended",
   "moves toward it" in _lf)
ok("fully_preserved, same reasoning as frame 0", "fully_preserved" in _lf)
ok("and cites its own ordinal", "<Picture 3>" in _lf)
# PLAN is four chunks; find where a shot actually ends
_cl = cr._closers(PLAN["chunks"])
ok("the last chunk always closes a shot", len(PLAN["chunks"]) - 1 in _cl)
ok("and `last frame` defaults to the closers, not to `all`",
   cr.default_scope("last frame", PLAN["chunks"])
   == ",".join(str(i + 1) for i in _cl))
_mid = [i for i in range(len(PLAN["chunks"])) if i not in _cl]
if _mid:
    _r = run(role="last frame", chunks="all", chunk_index=_mid[0])
    ok("mid-shot it still RENDERS, unlike frame 0", _r[1] is True)
    ok("but says the chunk is continued", "is CONTINUED by the one after it"
       in _r[3])

print("\nboth spellings of a cut are read, because two sources use two keys")
# H3ChunkPlan writes `cut`; takeapi.plan_fields writes `opens_shot`. Reading
# one made a plan from the other look like a single unbroken shot.
ok("`cut` opens a shot", cr.opens_shot({"cut": True}, 2) is True)
ok("`opens_shot` does too", cr.opens_shot({"opens_shot": True}, 2) is True)
ok("neither does not", cr.opens_shot({"cut": False}, 2) is False)
ok("and chunk 1 always does", cr.opens_shot({}, 0) is True)
check("openers read either key",
      cr._openers([{}, {"opens_shot": True}, {}, {"cut": True}]), [0, 1, 3])

print("\nthe plain `reference` role says only how hard to hold")
_g = cr.retention_clause("reference", 2, "weak_reference", "image")
ok("default is the weakest marker", "weak_reference" in _g)
ok("it claims no frame", "frame 0" not in _g and "final frame" not in _g)
ok("and says nothing about framing or motion",
   "composition" not in _g and "motion" not in _g)
ok("cited as a Picture for an image", "<Picture 2>" in _g)
_gv = cr.retention_clause("reference", 1, "attribute_transfer", "video")
ok("and as a Video for a clip", "<Video 1>" in _gv)
ok("the marker the caller chose is the one that lands",
   "attribute_transfer" in _gv)
ok("with the `how` the model needs after it", " - " in _gv)
ok("an unknown marker falls back rather than reaching the prompt",
   "weak_reference" in cr.retention_clause("reference", 1, "nonsense"))
ok("no depth: it is not a composition reference",
   cr.depth_for_role("reference") is False)
ok("and no depth for a last frame either",
   cr.depth_for_role("last frame") is False)
# A SINGLE-KIND ROLE MUST IGNORE `kind`. Reading the kind first made a motion
# reference cite <Picture N> the moment a default came through.
ok("kind cannot override a role that fixes its own tag",
   "<Video 2>" in cr.retention_clause("composition and motion", 2,
                                      kind="image"))
ok("every marker in the UI's list is one the model knows",
   all(m in cr.MARKER_HOW for m in _pl2.VISUAL_MARKERS))
ok("the new roles are offered for the right kinds",
   "last frame" in cr.ROLES_FOR["image"]
   and "reference" in cr.ROLES_FOR["image"]
   and "reference" in cr.ROLES_FOR["video"]
   and "last frame" not in cr.ROLES_FOR["video"])

print("\nand the UI offers exactly what the node accepts")
# THE VOCABULARY LIVED IN THREE PLACES ONCE AND TWO DISAGREED, so picking a
# marker in the UI raised an error naming a list the UI never showed. These
# two lists are typed separately -- one in Python, one in JavaScript -- so
# this is the only thing stopping them drifting apart again.
import os as _os
import re as _re
_ui = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "ui", "index.html"), encoding="utf-8").read()
_mk = _re.search(r"const VISUAL_MARKERS = \[(.*?)\];", _ui, _re.S)
ok("the UI declares a visual marker list", _mk is not None)
if _mk:
    _ui_markers = _re.findall(r'\["([a-z_]+)"', _mk.group(1))
    check("every marker the UI offers is one the node knows",
          [m for m in _ui_markers if m not in cr.MARKER_HOW], [])
    check("and it offers all of them",
          sorted(_ui_markers), sorted(cr.MARKER_HOW))
_rr = _re.search(r"const REF_ROLES = \{(.*?)\n\};", _ui, _re.S)
ok("the UI declares a role list", _rr is not None)
if _rr:
    # SPLIT ON THE KEYS rather than matching each block's end. The last key
    # has no following key to stop at, so a lookahead for one silently found
    # nothing and reported "the UI lists no video roles" -- a test failing on
    # its own regex, which is worse than no test.
    _parts = _re.split(r"\n\s*(image|video):\s*", _rr.group(1))
    _blocks = dict(zip(_parts[1::2], _parts[2::2]))
    for _kind in ("image", "video"):
        if _kind not in _blocks:
            ok(f"the UI lists {_kind} roles", False)
            continue
        _got = _re.findall(r'\["([a-z0-9 ]+)"', _blocks[_kind])
        # ROLES_OFFERED, not ROLES_FOR: `identity` is REACHABLE on the gated
        # path -- a cast member scoped to part of the take goes through this
        # gate -- but it is not offered as an additional reference, because a
        # person who should be recognised is a cast member.
        check(f"UI {_kind} roles are the node's",
              sorted(_got), sorted(cr.ROLES_OFFERED[_kind]))

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("chunk ref: scoping and the retention clause both hold")
