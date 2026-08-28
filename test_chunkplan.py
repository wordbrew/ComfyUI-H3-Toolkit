"""Chunk planner. No torch, no ComfyUI."""

import sys

from chunkplan import (find_cuts, latent_frames, legal_run, on_both_clocks, plan,
                       ref_share, describe, tokens_per_frame, video_tokens)

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

# --- the run must EQUAL what the chunk hands out ---------------------------- #
# This is the one that bit. `run` was rounded up while start/end were left
# alone, so a 40-frame tail asked for 56 generated frames against 40 source
# frames and the mask pin refused to resample them. The old assertion here was
# `run >= length`, which is exactly what let it through.
c, i = plan(200, 90, "scene", cuts=[85])
ok("run equals the frames handed out", all(x["run"] == x["length"] for x in c))
ok("runs are legal", all(legal_run(x["run"]) == x["run"] for x in c))

bad = []
for total in (39, 40, 90, 100, 199, 200, 201, 400, 447, 1200, 1201):
    for cf in (39, 56, 90, 100, 141):
        c, i = plan(total, cf)
        for x in c:
            if x["run"] != x["length"] or legal_run(x["run"]) != x["run"]:
                bad.append(f"{total}/{cf}: {x['length']}f handed out, run {x['run']}")
            if x["start"] < 0:
                bad.append(f"{total}/{cf}: chunk starts before the clip")
        # every kept frame exactly once, in order, from 0
        kept = [f for x in c for f in range(x["keep_from"], x["end"])]
        if kept != list(range(len(kept))):
            bad.append(f"{total}/{cf}: coverage is not contiguous")
        if len(kept) > total:
            bad.append(f"{total}/{cf}: kept {len(kept)} of {total}")
ok("legal run and exact coverage across 55 clip/chunk combinations", not bad)
fails.extend(bad[:6])

# V2V: a clip shorter than one legal run cannot be extended from footage that
# does not exist, so it trims down and says how many frames went
c, i = plan(100, 90)
check("short clip trims to a legal run", (c[0]["start"], c[0]["end"]), (0, 90))
ok("and reports the dropped frames",
   any("10 frame(s) dropped" in n for n in i["notes"]))

# FRESH GENERATION has nothing to run out of, so the same clip grows forward
c, i = plan(100, 90, grow_tail=True)
check("fresh generation grows instead of trimming",
      (c[0]["start"], c[0]["end"], c[0]["run"]), (0, 107, 107))
ok("and says nothing was dropped", any("grew to 107" in n for n in i["notes"]))
check("150 asked -> 158, not 141",
      plan(150, 141, grow_tail=True)[0][0]["run"], 158)
# growing is ONLY for the case with nothing behind it; a tail that can reach
# back is exact either way and must not overshoot
c, _ = plan(600, 141, grow_tail=True)
check("a reachable tail still lands exactly", sum(x["end"] - x["keep_from"]
                                                  for x in c), 600)
bad2 = []
for total in (39, 100, 150, 200, 564, 601, 1000):
    for cf in (39, 90, 141):
        c, _ = plan(total, cf, grow_tail=True)
        if any(x["run"] != x["length"] or x["run"] % 17 != 5 for x in c):
            bad2.append(f"{total}/{cf}")
        kept = [f for x in c for f in range(x["keep_from"], x["end"])]
        if kept != list(range(len(kept))) or len(kept) < total:
            bad2.append(f"{total}/{cf} coverage")
ok("growing never drops a frame and never breaks a run", not bad2)
fails.extend(bad2[:4])

# min_chunk is a threshold for becoming a chunk, so it snaps like the size does
c, i = plan(600, 90, min_chunk=50)
ok("min_chunk snaps to a legal run",
   any("min_chunk 50 is not a legal run" in n for n in i["notes"]))

# chunk_frames is snapped first, so every full part is legal by construction and
# only a tail can ever need correcting
c, i = plan(1200, 100)
ok("chunk_frames snaps to a legal run", all(x["run"] % 17 == 5 for x in c))
ok("and says so", any("not a legal run" in n for n in i["notes"]))
over = [x for x in c if x["keep_from"] > x["start"]]
check("only one chunk overlaps", len(over), 1)
check("and it is the last one", over[0]["end"], c[-1]["end"])

# --- a generated-audio tail prefers a run on BOTH clocks -------------------- #
# 300 frames at 141 with context 39 leaves a 96-frame tail. It rounds up to 107,
# which is a legal video run but does not divide by 3, so it rounds against the
# 40 Hz audio latent. Preserved audio absorbs that; GENERATED audio accumulates
# it down the chain, so the tail reaches back to 141 instead.
# generated_audio OFF: the tail is allowed to stay off the audio grid, so it
# takes 107 rather than reaching for 141. (grow_tail is on here, so this is a
# fresh generation that simply is not being held to the audio clock -- it is
# not the V2V path, which cannot grow forward at all.)
c, i = plan(300, 141, "fixed", context=39, grow_tail=True)
check("without the audio-grid preference the tail stays at 107",
      (c[-1]["start"], c[-1]["keep_from"], c[-1]["run"]), (204, 243, 107))
ok("but its pin still reaches the cut",
   c[-1]["start"] + c[-1]["pin"] >= c[-1]["keep_from"])
ok("and marks it off the grid", not c[-1]["both_clocks"])

c, i = plan(300, 141, "fixed", context=39, grow_tail=True, generated_audio=True)
# It used to reach BACK to 159 for this run. That put the cut 45 frames past
# where the 39-frame pin holds, and the seam there measured 3.56x the clip's
# median frame delta. It grows FORWARD now: same 141-frame both-clocks run, but
# starting where the pin can reach it.
check("generated audio takes a shared run its pin can reach",
      (c[-1]["start"], c[-1]["keep_from"], c[-1]["run"]), (204, 243, 141))
ok("every chunk now lands on both clocks", all(x["both_clocks"] for x in c))
ok("and says why the tail moved",
   any("lands on both clocks" in n for n in i["notes"]))
# keep_from is ABSOLUTE and must not follow the start backwards: the chunk simply
# throws more of its head away
check("keep_from did not move", c[-1]["keep_from"], 243)
kept = [f for x in c for f in range(x["keep_from"], x["end"])]
check("coverage is still contiguous from 0", kept, list(range(len(kept))))
ok("and never SHORT of the ask — the tail overshoots and Close trims the end",
   len(kept) >= 300)
# the "preserved audio absorbs it" note is a V2V fact and must not be printed on
# a chain that generates its own audio
c2, i2 = plan(300, 141, "scene", cuts=[47], context=39, generated_audio=True)
ok("no preserved-audio reassurance when the audio is generated",
   not any("PRESERVED" in n for n in i2["notes"]))

# it never reaches back past the start of its own shot. (It may still run PAST a
# cut: grow_tail extends a short shot forward, which is older behaviour and not
# what this flag touches.)
crossed = []
for cuts_at in ([150], [90, 200], [47, 120, 260]):
    for ga in (False, True):
        c, _ = plan(300, 141, "scene", cuts=cuts_at, context=39, grow_tail=True,
                    generated_audio=ga)
        firsts = {}
        for x in c:
            firsts.setdefault(x["shot"], x["start"])
            if x["start"] < firsts[x["shot"]] or x["start"] < 0:
                crossed.append(f"{cuts_at}/{ga}: chunk reached back out of its shot")
ok("back-extension never leaves its own shot", not crossed)

# the invariants, with the flag ON, across the same spread the V2V sweep uses
bad3 = []
for total in (39, 100, 150, 200, 300, 400, 564, 601, 1000, 1201):
    for cf in (39, 90, 141, 192):
        for ctx in (0, 22, 39):
            c, _ = plan(total, cf, context=ctx, grow_tail=True,
                        generated_audio=True)
            for x in c:
                if x["run"] != x["length"]:
                    bad3.append(f"{total}/{cf}/{ctx}: run {x['run']} != "
                                f"length {x['length']}")
                if legal_run(x["run"]) != x["run"]:
                    bad3.append(f"{total}/{cf}/{ctx}: run {x['run']} is not 17n+5")
                if x["start"] < 0:
                    bad3.append(f"{total}/{cf}/{ctx}: starts before the clip")
                if x["keep_from"] < x["start"] or x["keep_from"] > x["end"]:
                    bad3.append(f"{total}/{cf}/{ctx}: keep_from escaped the chunk")
            kept = [f for x in c for f in range(x["keep_from"], x["end"])]
            if kept != list(range(len(kept))) or len(kept) < total:
                bad3.append(f"{total}/{cf}/{ctx}: coverage is not exact")
ok("generated_audio keeps every invariant across 120 combinations", not bad3)
fails.extend(bad3[:6])

# and it is OFF by default, so no plan that already exists moves
moved = []
for total in (100, 200, 300, 564, 1000):
    for cf in (39, 90, 141):
        for ctx in (0, 22, 39):
            if plan(total, cf, context=ctx, grow_tail=True)[0] != \
                    plan(total, cf, context=ctx, grow_tail=True,
                         generated_audio=False)[0]:
                moved.append(f"{total}/{cf}/{ctx}")
ok("generated_audio defaults to off and changes nothing", not moved)

# --- the identity lever ---------------------------------------------------- #
r = 3 * 676  # three 832px references
share = {n: round(ref_share(n, r, 640, 1120) * 100, 1) for n in (39, 90, 294)}
check("ref share by chunk length", share, {39: 19.4, 90: 9.7, 294: 3.2})
ok("shorter chunks hold identity better", share[39] > share[90] > share[294])

# --- the cost column ------------------------------------------------------- #
# (w/32)*(h/32) per latent frame, and a run's latent frames under (1,4,4,4,4)
check("tokens per frame 640x1120", tokens_per_frame(640, 1120), 20 * 35)
check("tokens per frame 768x1344", tokens_per_frame(768, 1344), 24 * 42)
check("latent frames 90", latent_frames(90), 27)
check("latent frames 5", latent_frames(5), 2)
check("video tokens 90f at 640x1120", video_tokens(90, 640, 1120), 27 * 700)
# doubling the chunk length roughly doubles the tokens, which is why the same
# reference set is worth half as much in it
ok("cost tracks length", video_tokens(192, 640, 1120) > 2 * video_tokens(90, 640, 1120) * 0.9)

# --- the report renders ---------------------------------------------------- #
c, i = plan(1440, 90, "scene", cuts=[124, 400])
text = describe(c, i, render=(640, 1120), ref_tokens=r)
ok("report names the mode", "chunk_mode: scene" in text)
ok("report lists cuts", "cuts detected at: 124, 400" in text)
ok("report flags a track restart", "[track restarts]" in text)
# both numbers that a chunk length decides, on every chunk line
ok("report prints the token cost", "18,900 tokens" in text)
ok("report prints the reference share", "refs  9.7%" in text)
ok("report prints the render size and its per-frame cost",
   "render 640x1120" in text and "700 tokens per latent frame" in text)
ok("report says the references were counted", "2,028 reference tokens" in text)
# the flags all survive alongside the new columns
ok("clock flag still on the chunk line",
   all(("both clocks" in ln or "OFF grid" in ln)
       for ln in text.splitlines() if ln.strip().startswith(("01 ", "02 ", "03 "))))
ok("overlap note survives", "dropped at the join" in describe(
    plan(1200, 100)[0], plan(1200, 100)[1], render=(640, 1120), ref_tokens=r))

# where the size came from is part of the report: a size derived from the clip
# and a size typed into a widget are different claims
ok("names source_images", "render 896x672 (from source_images)"
   in describe(c, i, render=(896, 672), render_from="from source_images"))
ok("names the widgets", "(from the widgets)"
   in describe(c, i, render=(640, 1120), render_from="from the widgets"))
# with no render size there is no cost column and no render line at all
plain = describe(c, i)
ok("no render size, no columns",
   "tokens" not in plain and "refs" not in plain and "render " not in plain)
# ref_tokens alone cannot produce a share -- it needs a canvas to be a share OF
ok("tokens without references still print",
   "tokens" in describe(c, i, render=(640, 1120)) and
   "refs" not in describe(c, i, render=(640, 1120)))
# columns line up: every chunk line puts "tokens" at the same offset
offsets = {ln.index("tokens") for ln in text.splitlines()
           if " frames " in ln and "tokens" in ln}
check("token column is aligned", len(offsets), 1)

# --- the pin has to REACH the cut ------------------------------------------- #
# Everything from `start` to `keep_from` is dropped at the join, but only the
# first `pin` frames of it are HELD. Reach back further than the pin covers and
# the rest runs unheld, drifts, and the join cuts to a chunk that has gone
# somewhere else. Measured 2026-08-28 on a 300-frame T2V chain: the chunk whose
# pin ended exactly on its cut seamed at 0.81x the clip's median frame delta,
# and the chunk reached back 84 frames behind a 39-frame pin seamed at 3.56x --
# the single largest step in the clip.
bad3 = []
for total in (150, 200, 300, 301, 400, 447, 600, 1000):
    for cf in (39, 90, 141, 192):
        for ctxv in (0, 1, 5, 22, 39):
            for fresh in (False, True):
                c, _ = plan(total, cf, context=ctxv, grow_tail=fresh,
                            generated_audio=fresh)
                for x in c:
                    # ctx 0 means no carry at all, so there is no pin to outrun
                    if x["pin"] and x["start"] + x["pin"] < x["keep_from"]:
                        bad3.append(f"{total}/{cf}/ctx{ctxv}/fresh{fresh}: pin "
                                    f"{x['pin']} at {x['start']} does not reach "
                                    f"{x['keep_from']}")
                    if x["run"] != x["length"] or legal_run(x["run"]) != x["run"]:
                        bad3.append(f"{total}/{cf}/ctx{ctxv}: run != length")
                kept = [f for x in c for f in range(x["keep_from"], x["end"])]
                if kept != list(range(len(kept))):
                    bad3.append(f"{total}/{cf}/ctx{ctxv}: coverage broken")
ok("every pin that exists reaches its own cut, across 320 combinations", not bad3)
fails.extend(bad3[:6])

# the exact case that was measured
c, i = plan(300, 141, context=39, grow_tail=True, generated_audio=True)
check("the tail starts where its pin can reach",
      (c[-1]["start"], c[-1]["keep_from"]), (204, 243))
check("and it is still a legal both-clocks run", c[-1]["run"], 141)
ok("it grew forward rather than back", c[-1]["end"] > 300)
# V2V cannot grow forward, so it trims and says how much went
c, i = plan(300, 141, context=39)
check("V2V trims instead", (c[-1]["start"], c[-1]["end"]), (204, 294))
ok("and says why", any("pin holds" in n for n in i["notes"]))

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("chunk planner: all checks pass")
