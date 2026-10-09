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

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("chunk ref: scoping and the retention clause both hold")
