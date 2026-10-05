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
