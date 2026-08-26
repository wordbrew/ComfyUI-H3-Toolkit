"""Chunk planner. No torch, no ComfyUI."""

import sys

from chunkplan import (find_cuts, legal_run, on_both_clocks, plan, ref_share,
                       describe)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)


# --- the two clocks -------------------------------------------------------- #
check("legal up 47", legal_run(47), 56)
check("legal up 90", legal_run(90), 90)
check("legal down 47", legal_run(47, "down"), 39)
check("legal floor", legal_run(1), 5)
# the intersection of the two grids: every third video run, 51 apart
both = [n for n in range(5, 400) if on_both_clocks(n)]
check("both clocks", both, [39, 90, 141, 192, 243, 294, 345, 396])
ok("22 is a legal run but off the audio grid",
   legal_run(22) == 22 and not on_both_clocks(22))
ok("362 is legal video, off audio", legal_run(362) == 362 and not on_both_clocks(362))

# --- cut detection --------------------------------------------------------- #
# index 0 can never be a cut: there is no frame before it
check("no cut at 0", find_cuts([9.9, 0.1, 0.1], 1.0), [])
check("cuts found", find_cuts([0, .1, .1, 5.0, .1, 4.0], 1.0), [3, 5])

# --- fixed mode ------------------------------------------------------------ #
c, i = plan(1440, 90, "fixed")
check("fixed count", len(c), 16)
check("fixed covers clip", (c[0]["start"], c[-1]["end"]), (0, 1440))
ok("fixed is contiguous", all(c[k]["end"] == c[k + 1]["start"] for k in range(15)))
ok("fixed all on both clocks", all(x["both_clocks"] for x in c))
ok("fixed seeds after the first", not c[0]["seed_mask"] and all(x["seed_mask"] for x in c[1:]))

# An indivisible clip absorbs its tail into the last chunk rather than leaving a
# runt, and rather than splitting evenly. Even splitting looks tidier and
# generates MORE: 1000 at 90 would become 11x91, each padding to 107 = 1177
# frames generated for 1000 of content.
c, i = plan(1000, 90, "fixed")
check("tail absorbed, still covers clip", (c[0]["start"], c[-1]["end"]), (0, 1000))
ok("no runt", all(x["length"] >= 39 for x in c))
generated = sum(x["run"] for x in c)
ok(f"padding waste stays small: {generated} generated for 1000",
   generated <= 1000 * 1.05)

# a shot barely over the chunk size is taken WHOLE rather than halved: 124 at
# size 90 splits 62/62 -> 73/73 = 146 generated, against 124 taken in one pass
c, i = plan(124, 90, "fixed")
check("124 taken whole", [(x["length"], x["run"]) for x in c], [(124, 124)])

# --- scene mode ------------------------------------------------------------ #
cuts = [124, 400, 900]
c, i = plan(1440, 90, "scene", cuts=cuts)
check("shots", i["shots"], 4)
ok("every chunk sits inside one shot",
   all(not (x["start"] < k < x["end"]) for x in c for k in cuts))
ok("chunk starts land on cuts or splits",
   all(x["start"] in cuts or x["start"] == 0 or x["part"] for x in c))
# the first chunk of each shot must NOT seed its track from the previous shot
firsts = [x for x in c if x["part"] is None or x["part"][0] == 1]
ok("no track seeding across a cut", all(not x["seed_mask"] for x in firsts))
ok("seeding does happen within a split shot",
   any(x["seed_mask"] for x in c))

# short shots merge rather than producing an ungeneratable chunk
c, i = plan(600, 90, "scene", cuts=[10, 20, 30], min_chunk=39)
ok("tiny shots merged", all(x["length"] >= 39 for x in c))
check("merged coverage", (c[0]["start"], c[-1]["end"]), (0, 600))

# scene mode with no cuts degrades to fixed and says so
c, i = plan(500, 90, "scene", cuts=[])
check("degrades to fixed", i["mode"], "fixed")
ok("and says why", any("fell back to fixed" in n for n in i["notes"]))

# --- padding --------------------------------------------------------------- #
c, i = plan(200, 90, "scene", cuts=[85])
ok("short shots pad UP to a legal run", all(x["run"] >= x["length"] for x in c))
ok("runs are legal", all(legal_run(x["run"]) == x["run"] for x in c))

# --- the identity lever ---------------------------------------------------- #
r = 3 * 676  # three 832px references
share = {n: round(ref_share(n, r, 640, 1120) * 100, 1) for n in (39, 90, 294)}
check("ref share by chunk length", share, {39: 19.4, 90: 9.7, 294: 3.2})
ok("shorter chunks hold identity better", share[39] > share[90] > share[294])

# --- the report renders ---------------------------------------------------- #
c, i = plan(1440, 90, "scene", cuts=[124, 400])
text = describe(c, i, render=(640, 1120), ref_tokens=r)
ok("report names the mode", "chunk_mode: scene" in text)
ok("report lists cuts", "cuts detected at: 124, 400" in text)
ok("report flags a track restart", "[track restarts]" in text)

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("chunk planner: all checks pass")
