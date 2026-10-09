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

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("chunk ref: scoping and the retention clause both hold")
