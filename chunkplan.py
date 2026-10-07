"""Where to cut a long clip into chunks. Torch-free, so it can be tested.

The node in `longform.py` does the frame differencing and owns the tensors; the
arithmetic lives here because it is the part with rules worth checking.

TWO MODES, AND THE TRADE BETWEEN THEM

  `fixed`  every chunk is `chunk_frames`. Uniform cost, and every chunk can sit
           on both clocks. A chunk may span a scene change, which means ONE
           prompt describing TWO scenes -- prompts describing the wrong thing
           have cost us more than any other single cause.

  `scene`  chunks end at cuts, so each chunk is one continuous shot and the
           prompt always matches its content. Shots are whatever length they
           are, so most chunks land off the frame grid and get padded up.

WHY `scene` IS CHEAP HERE AND WOULD NOT BE IN A GENERATIVE CHAIN
  Off-grid lengths matter when audio is GENERATED: each link rounds against the
  40 Hz audio latent and the error accumulates down the chain. A V2V swap
  PRESERVES the source audio -- sliced from the original and reassembled at
  exact integer sample boundaries -- so an odd chunk costs a rounding at that
  chunk and nothing downstream. That is what makes cutting at scene boundaries
  affordable, and it is the opposite of the long-form case.

WHY CHUNK LENGTH IS ALSO AN IDENTITY LEVER
  A reference image is a FIXED token count while the target grows with duration,
  so a reference's share of the picture tokens falls as chunks get longer. At
  640x1120 with three 832px references: 39f -> 19.4%, 90f -> 9.7%, 294f -> 3.2%.
  The 300-frame render that lost its replacement character's hair colour sat at
  3.2%. Shorter chunks are not only cheaper, they hold identity better.
"""

FPS = 24
AUDIO_LATENT_HZ = 40


def legal_run(frames, direction="up"):
    """Nearest legal video run (17n+5). Never returns less than 5."""
    n = max(5, int(frames))
    if direction == "down":
        while n % 17 != 5 and n > 5:
            n -= 1
        return max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def on_both_clocks(frames):
    """True when a run lands exactly on the video grid AND the audio grid.

    Video runs are 17n+5. Audio is a 40 Hz latent against 24 fps video, so a run
    lands on a whole audio step only when frames*40 is divisible by 24 -- i.e.
    when the frame count divides by 3. The intersection is every third video
    run, 51 apart: 39, 90, 141, 192, 243, 294, 345.
    """
    n = int(frames)
    return n > 5 and (n - 5) % 17 == 0 and (n * AUDIO_LATENT_HZ) % FPS == 0


def find_cuts(deltas, threshold):
    """Frame indices where a hard cut starts, from per-frame difference scores.

    `deltas[i]` is how different frame i is from frame i-1, so a returned index
    is the FIRST frame of the new shot. Index 0 is never a cut -- there is no
    frame before it and the clip already starts there.

    Hard cuts only. A dissolve spreads its change across many frames and never
    trips a single-frame threshold; the caller says so rather than pretending
    otherwise.
    """
    return [i for i, d in enumerate(deltas) if i > 0 and float(d) >= float(threshold)]


CONTEXT_GRID = (39, 22, 5, 1)


def snap_context(frames):
    """Nearest usable context length at or below `frames`, or 0 for none.

    Only 39, 22, 5 and 1 pixel frames encode to distinct VAE runs. An off-grid
    count snaps DOWN to the next one and then covers the FIRST frames of what it
    was given rather than the last, so the pin ends early and the join jumps --
    30 silently becomes the wrong 22. Snapping here means the number the plan
    reports is the number that actually gets pinned.
    """
    n = int(frames)
    if n <= 0:
        return 0
    for g in CONTEXT_GRID:
        if n >= g:
            return g
    return 0


# ---------------------------------------------------------------- continuity
# HOW ONE CHUNK FOLLOWS THE ONE BEFORE. We had two options, `pin` and `pin 0`,
# which is "the shot continues" and "hard cut" -- and nothing between. This
# vocabulary is adopted from WAS Node Suite's MiniMax H3 Conditioning
# (WASasquatch/was-node-suite-comfyui, MIT), whose timeline names eight; two of
# them answer problems recorded in this repo and had no expression here at all.
#
# SUPPORTED means this planner emits a pin and an audio decision the existing
# nodes already honour. DECLARED means the vocabulary carries it so a plan can
# say what it wants, and `notes` says plainly that the conditioning for it is
# not wired -- better than a name that silently does something else.
CONTINUITY = {
    # name              pin frames          audio     supported
    "carry":           {"pin": "context",  "audio": "carry", "ok": True,
                        "what": "the shot goes on from its last frames"},
    "cut":             {"pin": 0,          "audio": "fresh", "ok": True,
                        "what": "a new scene, nothing carried"},
    "handoff":         {"pin": 1,          "audio": "fresh", "ok": True,
                        "what": "a new shot opening ON the last frame"},
    # THE ONE OUR OWN NOTES ASKED FOR. Generated audio restarts at every link,
    # so a cut in the picture takes the music with it. Carrying the audio across
    # a picture cut is a separate decision from carrying the picture, and the
    # pin machinery already keeps video and audio masks apart.
    "audio_carry":     {"pin": 0,          "audio": "carry", "ok": True,
                        "what": "the picture cuts, the sound continues"},
    # DECLARED ONLY -- each needs a conditioning path this pack does not build.
    "refresh":         {"pin": "context",  "audio": "carry", "ok": False,
                        "what": "the same shot with the carried frames re-noised",
                        "why": "a fractional denoise mask on the carried rows, "
                               "and on 0.35+ a fractional mask scales the "
                               "model's velocity so those rows never finish "
                               "denoising -- see feather_warning()"},
    "reference_video": {"pin": 0,          "audio": "fresh", "ok": False,
                        "what": "a new shot REFERENCING the last frames",
                        "why": "the tail has to enter as a reference VIDEO, not "
                               "a pin; motion and appearance are the same tokens "
                               "on a ref video, which is its own measured trap"},
    # WIRED. H3ChunkRefSample takes evenly spaced stills from the previous
    # chunk's decoded output and hands them to a reference slot, firing only
    # where a plan asks for it. Appearance crosses the cut; motion does not.
    "reference_sample":{"pin": 0,          "audio": "fresh", "ok": True,
                        "what": "a new shot referencing stills from the clip "
                                "before it — the cast crosses the cut, the "
                                "motion does not",
                        "needs": "H3 Chunk Ref Sample wired from H3 Chunk Open's "
                                 "`prev_frames` into a reference slot"},
    "audio_reference": {"pin": 0,          "audio": "carry", "ok": False,
                        "what": "sound continues and the last frames are referenced",
                        "why": "reference_video plus audio_carry"},
}
DEFAULT_CONTINUITY = "carry"
#: Picture does not continue across these, whatever the audio does.
CUT_LIKE = tuple(k for k, v in CONTINUITY.items() if v["pin"] != "context")


def continuity_pin(name, context):
    """-> (pin frames, audio decision, supported). Unknown names fall back."""
    spec = CONTINUITY.get(name) or CONTINUITY[DEFAULT_CONTINUITY]
    pin = spec["pin"]
    return (int(context) if pin == "context" else int(pin),
            spec["audio"], bool(spec["ok"]))


LEGAL_RUNS = tuple(17 * k + 5 for k in range(0, 40))
# legal AND exact on the 40 Hz audio grid (frames * 40 / 24 is a whole number)
AV_EXACT_RUNS = tuple(r for r in LEGAL_RUNS if (r * 40) % 24 == 0)


def scene_plan(scenes, context=39, fps=24.0):
    """A LIST OF SCENES, each with its own length and transition.

    `count_plan` answers "n chunks of one size", which is right for a single
    continuous take and wrong for a sequence of shots -- a four-second insert
    and a twelve-second master are the same film. A scene here is
    {frames, continuity, beat}, and every scene sets its own length.

    Each scene RENDERS `frames`. What it DELIVERS depends on the transition
    into it: a carry overlaps the scene before by its pin and that overlap is
    dropped at the join, so it delivers `frames - pin`. A cut overlaps nothing
    and delivers all of it. This is the same arithmetic count_plan does, per
    scene instead of once.

    -> {scenes: [...], total_frames, total_seconds, notes}
    """
    ctx = int(context)
    out, notes = [], []
    at = 0
    for i, sc in enumerate(scenes or []):
        want = int(sc.get("frames") or 0)
        frames = legal_run(want, "nearest") if want else legal_run(141, "nearest")
        if want and frames != want:
            notes.append(f"scene {i + 1}: {want} is not a legal run — rendered "
                         f"at {frames} ({frames / fps:.2f}s)")
        name = (sc.get("continuity") if i else None) or (None if i == 0 else "carry")
        # A SCENE MAY OVERRIDE THE CARRY. `context` is the default for the whole
        # take, which is right most of the time -- but a scene continuing from a
        # fast-moving shot has no reason to pin the same number of frames as one
        # continuing from a held frame. An absent or None `pin` takes the global.
        here_ctx = sc.get("pin")
        here_ctx = ctx if here_ctx in (None, "") else int(here_ctx)
        pin, audio, wired = (0, "fresh", True) if i == 0 else \
            continuity_pin(name, here_ctx)
        pin = min(pin, max(0, frames - 1))      # a pin cannot eat the whole scene
        delivered = frames - pin
        if not wired:
            notes.append(f"scene {i + 1}: {name!r} is named but not wired — "
                         f"{CONTINUITY[name].get('why', '')}")
        out.append({
            "index": i, "number": i + 1,
            "frames": frames,                   # what the sampler renders
            "pin": pin,                         # carried in from the scene before
            "delivered": delivered,             # what reaches the finished clip
            "start": at, "end": at + delivered,
            "start_seconds": round(at / fps, 3),
            "end_seconds": round((at + delivered) / fps, 3),
            "seconds": round(delivered / fps, 3),
            "continuity": name,
            "pin_default": ctx,
            "pin_overridden": sc.get("pin") not in (None, "") and i > 0,
            "carry_audio": audio == "carry" and i > 0,
            "continuity_wired": wired,
            "both_clocks": on_both_clocks(frames),
            "beat": sc.get("beat") or None,
        })
        at += delivered
    off = [c["number"] for c in out if not c["both_clocks"]]
    if off:
        notes.append(f"scene(s) {', '.join(map(str, off))} are off the 40 Hz "
                     f"audio grid, so each join there rounds the audio by up to "
                     f"half a tick; the exact runs are "
                     f"{', '.join(map(str, AV_EXACT_RUNS[:6]))}...")
    return {"scenes": out, "total_frames": at,
            "total_seconds": round(at / fps, 3), "notes": notes}


def total_for_count(chunk_frames, chunk_count, context):
    """Frames delivered by `chunk_count` chunks of `chunk_frames`, carrying `context`.

    THE ASYMMETRY IS ARITHMETIC, NOT A BUG. Chunk 1 keeps everything it renders;
    every later chunk drops the carried handle. So the delivered lengths are
    `chunk_frames` then `chunk_frames - context`, and they can NEVER be equal:
    that would need `A = B - C` with all three on the 17n+5 grid, and every legal
    run is 5 mod 17, so B-C is 0 mod 17 and A is 5. No solutions. Report it
    rather than pretending otherwise.
    """
    cf, n, ctx = int(chunk_frames), max(1, int(chunk_count)), int(context)
    return cf + (n - 1) * max(1, cf - ctx)


def count_plan(chunk_frames, chunk_count, context):
    """-> (total_frames, first_delivered, rest_delivered, notes)

    The single authority for "how long will this be". The NODE calls it and the
    /chunk/plan HTTP route calls it, so a front end previewing a take and the
    graph rendering it cannot disagree -- two implementations of one rule is the
    failure this pack keeps paying for.
    """
    cf, n, ctx = int(chunk_frames), max(1, int(chunk_count)), int(context)
    notes = []
    if cf not in LEGAL_RUNS:
        near = min(LEGAL_RUNS, key=lambda x: abs(x - cf))
        notes.append(f"chunk_frames {cf} is not a legal run (17n+5); nearest is {near}")
    if ctx and ctx not in LEGAL_RUNS:
        notes.append(f"context {ctx} is not a legal run (17n+5) — the carry moves "
                     f"content between clips and only 17n+5 lands in phase")
    if ctx >= cf:
        notes.append(f"context {ctx} is not smaller than chunk_frames {cf}; "
                     f"nothing new would be generated")
    if cf not in AV_EXACT_RUNS:
        notes.append(f"{cf} frames is {cf * 40 / 24:.2f} audio ticks — off the "
                     f"40 Hz grid. Exact sizes: "
                     f"{', '.join(str(r) for r in AV_EXACT_RUNS if 39 <= r <= 400)}")
    first = cf
    rest = max(1, cf - ctx)
    return total_for_count(cf, n, ctx), first, rest, notes


def describe_count_plan(chunk_frames, chunk_count, context, fps=24.0):
    """The human-readable answer to 'how long will this be', before anything runs."""
    total, first, rest, notes = count_plan(chunk_frames, chunk_count, context)
    n = max(1, int(chunk_count))
    L = [f"{n} chunk(s) x {int(chunk_frames)} frames, carrying {int(context)}"]
    if n > 1:
        L.append(f"  chunk 1 delivers {first} ({first / fps:.2f}s), "
                 f"chunks 2-{n} deliver {rest} ({rest / fps:.2f}s) each")
    L.append(f"  TOTAL {total} frames = {total / fps:.2f}s")
    L.append(f"  write {n} beat(s) — one per chunk")
    alts = []
    for cand in LEGAL_RUNS:
        if cand == int(chunk_frames) or not (39 <= cand <= 400):
            continue
        t = total_for_count(cand, n, context)
        alts.append((abs(t - total), cand, t))
    for _, cand, t in sorted(alts)[:2]:
        L.append(f"  alternative: {cand} frames -> {t} total = {t / fps:.2f}s")
    L += [f"  {x}" for x in notes]
    return "\n".join(L)


def plan(total_frames, chunk_frames=90, mode="fixed", cuts=None, min_chunk=39,
         context=0, grow_tail=False, *, generated_audio=False, continuity=None):
    """-> (chunks, info)

    Each chunk is a dict:
      start, end     source frame range, end exclusive
      length         end - start, the SOURCE frames this chunk covers
      run            the legal run generated. ALWAYS equals `length`: a chunk
                     hands out `length` source frames and gets `run` back, so
                     if they differ the mask pin has nothing to pin to.
      keep_from      where this chunk's NEW content begins. Greater than `start`
                     when the chunk was extended backwards to reach a legal run;
                     the join drops frames before it, so the overlap is
                     regenerated but never appears twice.
      shot           index of the shot it belongs to, or None in fixed mode
      part           (i, n) when a shot was split, else None
      both_clocks    whether `run` lands on the video and audio grids
      seed_mask      whether SAM3 may be seeded from the previous chunk

    `seed_mask` is the correction that came out of the rapid-cuts problem.
    Seeding a track with the previous chunk's final mask keeps the boundary from
    jumping -- but ACROSS A CUT it points SAM3 at where the subject stood in a
    different shot, which is worse than re-detecting. So it is False on the first
    chunk of every shot and True only within a shot.

    The identity carry has the opposite rule and is not decided here: a
    reference image is framing-agnostic, so it crosses cuts safely.

    `generated_audio` says the audio is being GENERATED rather than preserved,
    which changes what an off-grid chunk costs. A V2V swap slices the source
    audio and reassembles it at exact integer sample boundaries, so an off-grid
    chunk costs one rounding there and nothing downstream. The ref2va model
    generates its own audio, and then every link rounds against the 40 Hz latent
    and the error accumulates down the chain -- so a tail that has to reach back
    anyway reaches back far enough to land on BOTH clocks. It is deliberately a
    separate flag from `grow_tail`: they share a trigger today (both mean "no
    source clip"), but one is about inventing frames and the other about the
    audio clock, and collapsing them would hide that.
    """
    total = int(total_frames)
    # Snap the chunk size to a legal run FIRST. Every full part is then legal by
    # construction and only a tail can be short, which is the one case that
    # needs correcting below.
    asked = max(5, int(chunk_frames))
    size = legal_run(asked, "up")
    notes = []
    if size != asked:
        notes.append(f"chunk_frames {asked} is not a legal run — using {size} "
                     f"(runs are 17n+5)")

    # min_chunk gates whether a short shot is merged or becomes a chunk of its
    # own, and a chunk of its own has to be a legal run -- so the threshold is
    # snapped for the same reason the size is.
    asked_min = max(5, int(min_chunk))
    min_chunk = legal_run(asked_min, "up")
    if min_chunk != asked_min:
        notes.append(f"min_chunk {asked_min} is not a legal run — using {min_chunk}")

    ctx = snap_context(context)
    if ctx != int(context or 0):
        notes.append(f"context {context} is not a usable pin length — using {ctx} "
                     f"(only 39, 22, 5 and 1 encode distinctly)")
    if ctx >= size:
        notes.append(f"context {ctx} does not fit inside a {size}-frame chunk — "
                     f"dropped to 0; raise chunk_frames to carry context")
        ctx = 0

    if total <= 0:
        return [], {"chunks": 0, "mode": mode, "notes": ["empty clip"]}

    if mode == "scene" and not cuts:
        notes.append("scene mode asked for but no cuts were found — fell back to "
                     "fixed. Hard cuts only; a dissolve will not trip the "
                     "threshold.")
        mode = "fixed"

    # THE MIRROR CASE, which had no warning at all. Cuts only shape the plan in
    # `scene` mode; a fixed plan ignores them, every chunk keeps its carry, and
    # the take comes back as one continuous shot. That cost a 59-second render:
    # cuts were typed, the report said "cuts detected at: 612, 1224", and the
    # output had no cut in it. Say so, and stop reporting them as applied.
    ignored_cuts = []
    if mode != "scene" and cuts:
        ignored_cuts = sorted(set(int(c) for c in cuts))
        notes.append(
            f"{len(ignored_cuts)} cut(s) were given but chunk_mode is "
            f"{mode!r}, which IGNORES them — every chunk keeps its carry and "
            f"the take is one continuous shot. Set chunk_mode to 'scene' to "
            f"make the chunks end at the cuts. Ignored: "
            + ", ".join(str(c) for c in ignored_cuts[:8])
            + ("..." if len(ignored_cuts) > 8 else ""))
        cuts = None

    if mode == "scene":
        bounds = [0] + [c for c in sorted(set(int(c) for c in cuts))
                        if 0 < c < total] + [total]
        shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        # A very short shot cannot be generated on its own -- the smallest legal
        # run is 5 and the smallest USEFUL one is min_chunk. Merge it forward
        # into the next shot, or backward if it is the last.
        merged, i = [], 0
        while i < len(shots):
            s, e = shots[i]
            while e - s < int(min_chunk) and i + 1 < len(shots):
                i += 1
                e = shots[i][1]
                notes.append(f"shot boundary at {s} merged — under {min_chunk} frames")
            if e - s < int(min_chunk) and merged:
                ps, _ = merged.pop()
                s = ps
                notes.append(f"final short shot merged backward")
            merged.append((s, e))
            i += 1
        shots = merged
    else:
        shots = [(0, total)]

    chunks = []
    for si, (s0, s1) in enumerate(shots):
        shot_len = s1 - s0
        # Greedy fill at `size`, absorbing a short tail into the last part.
        # An even split looks tidier and generates MORE: a 124-frame shot at
        # size 90 splits 62/62, each padding up to 73, so 146 frames are
        # generated for 124 of content. Taking 124 whole is one legal run, one
        # pass, and no seam. The tail is absorbed rather than left as a runt
        # because a runt would also carry a different reference share from
        # every other chunk.
        # With context, a chunk OVERLAPS its predecessor: it still spans `size`
        # frames but only advances `size - context`, and the frames it shares are
        # where the previous chunk's finished output gets pinned in. Without
        # context (0) the stride is the full size and chunks butt together.
        stride = max(5, size - ctx)
        parts, pos = [], s0
        while s1 - pos > size + int(min_chunk):
            parts.append((pos, pos + size))
            pos += stride
        parts.append((pos, s1))
        n_parts = len(parts)
        for p, (start, end) in enumerate(parts):
            # A chunk hands out `end - start` source frames and the model
            # generates `run` of them, so the two MUST be the same number. They
            # were not: `run` was rounded up and `start`/`end` left alone, and
            # every tail that did not happen to land on 17n+5 reached the mask
            # pin as (say) 39 source frames against a 56-frame latent, which it
            # correctly refused to resample.
            #
            # The fix is to move the chunk's START back so it covers a full
            # legal run, and remember where its NEW content begins. The overlap
            # is regenerated and then dropped at the join, so nothing is lost
            # and no frame appears twice.
            run = legal_run(end - start, "up")
            deficit = run - (end - start)
            # everything before keep_from is content the previous chunk already
            # delivered: the context overlap, plus whatever the tail reached back
            # for. All of it is regenerated and dropped at the join.
            keep_from = start + (ctx if p > 0 else 0)
            pin_here = ctx if p > 0 else 0
            # THE TRANSITION DECIDES THE PIN, where one was given. `continuity`
            # is indexed by JOIN: entry 0 is the transition into chunk 2, so the
            # first chunk never has one -- there is nothing behind it to follow
            # from. Anything not given stays on the position-derived default, so
            # a plan with no continuity at all behaves exactly as before.
            here_name = DEFAULT_CONTINUITY
            audio_here = "carry"
            if p > 0 or chunks:
                idx = len(chunks) - 1
                if continuity and 0 <= idx < len(continuity):
                    here_name = str(continuity[idx] or DEFAULT_CONTINUITY)
                    cpin, audio_here, okc = continuity_pin(here_name, ctx)
                    pin_here = cpin
                    keep_from = start + cpin
                    if not okc:
                        notes.append(
                            f"chunk {len(chunks) + 1} asks for {here_name!r}: "
                            f"{CONTINUITY[here_name]['what']} — NOT WIRED. "
                            f"{CONTINUITY[here_name].get('why', '')} Planned as "
                            f"pin {cpin}, so it renders as the nearest thing "
                            f"this pack can actually do.")
            else:
                audio_here = "fresh"
            # This chunk is already reaching back for a legal video run. When the
            # audio is GENERATED, reach a little further and land on the audio
            # grid too -- 300 frames at 141 with context 39 leaves a 96-frame
            # tail, which rounds to 107: a legal run, but 107 does not divide by
            # 3 so it rounds against the 40 Hz latent, and in a generated chain
            # that error accumulates. 141 is the next run on both clocks and the
            # extra head is thrown away at the join, so the only cost is compute.
            if deficit and generated_audio and not on_both_clocks(run):
                shared = run
                while not on_both_clocks(shared):
                    shared = legal_run(shared + 1, "up")
                extra = shared - (end - start)
                if start - extra >= s0:      # still inside this shot, never across a cut
                    notes.append(f"tail run {run} is a legal video run but off the "
                                 f"audio grid — extended to {shared}, which lands on "
                                 f"both clocks. The audio here is generated, so an "
                                 f"off-grid link's rounding would accumulate.")
                    run, deficit = shared, extra
            # THE PIN HAS TO REACH THE CUT. Everything from `start` to
            # `keep_from` is thrown away at the join, but only the first `pin`
            # frames of it are HELD -- the rest are generated with nothing
            # holding them, and by the cut point they have drifted somewhere
            # else. Reaching back further than the pin can cover is therefore
            # not free: it buys a legal run and pays for it with a visible jump.
            #
            # Measured 2026-08-28 on a 300-frame T2V chain. Chunk 2's pin ended
            # exactly on its cut and its seam was 0.81x the clip's MEDIAN
            # frame-to-frame delta -- invisible. Chunk 3 had been reached back 84
            # frames with a 39-frame pin, so 45 frames ran unheld, and its seam
            # was 3.56x the median and the single largest step in the whole clip.
            #
            # So a chunk that carries a pin does not move backwards at all. It
            # takes its legal run FORWARD instead, and only the no-pin case
            # (context 0) keeps the old behaviour, where reaching back is exact
            # and costs nothing.
            if deficit and pin_here and start + pin_here < keep_from + deficit:
                if grow_tail:
                    # generating: nothing to run out of, so overshoot the ask
                    notes.append(f"tail grew forward to {run} frames rather than "
                                 f"reaching back past its {pin_here}-frame pin — "
                                 f"a chunk cut where nothing is holding it jumps "
                                 f"(measured 3.6x the clip's median delta)")
                    end = start + run
                else:
                    # V2V: forward is past the end of the footage, so take the
                    # run BELOW and drop the remainder rather than cut at a
                    # frame the pin never reached
                    run = legal_run(end - start, "down")
                    dropped = (end - start) - run
                    end = start + run
                    notes.append(f"tail trimmed to {run} frames, {dropped} "
                                 f"dropped: reaching back for {run + dropped} "
                                 f"would have put the cut {dropped} frame(s) "
                                 f"past where the {pin_here}-frame pin holds")
            elif deficit and start - deficit >= s0:
                # no pin to outrun — reaching back is exact and lossless here.
                # keep_from is an ABSOLUTE frame index and does not move: the
                # chunk simply starts earlier and throws more of its head away.
                start -= deficit
            elif deficit and grow_tail:
                # FRESH GENERATION has no source to run out of, so a chunk with
                # nothing behind it grows FORWARD to the next legal run instead
                # of being trimmed. You asked for 150 and get 158, rather than
                # asking for 150 and getting 141.
                notes.append(f"{end - start} frames is not a legal run — grew to "
                             f"{run}, so nothing is dropped (generating, not "
                             f"cutting footage)")
                end = start + run
            elif deficit:
                # V2V: nothing behind it AND no footage past the end. Trim to the
                # run below rather than ask for frames that do not exist.
                run = legal_run(end - start, "down")
                dropped = (end - start) - run
                end = start + run
                notes.append(f"shot at {start} is {run + dropped} frames, shorter "
                             f"than a legal run — trimmed to {run}, "
                             f"{dropped} frame(s) dropped")
            length = end - start
            chunks.append({
                "start": start, "end": end, "length": length, "run": run,
                "keep_from": keep_from,
                # leading frames to overwrite with the PREVIOUS chunk's finished
                # output before anything is encoded. 0 on the first chunk of a
                # shot -- there is nothing behind it, and across a cut there is
                # nothing worth carrying.
                "pin": pin_here,
                # Does this chunk START a new shot? NOT the same question as
                # `pin == 0`: a continuous take planned with context 0 also pins
                # nothing, and there the previous tail is still worth handing on
                # as motion context. Only across a CUT is carrying it actively
                # wrong, and the runner used to do it regardless -- so a new shot
                # opened on the frame it was meant to be cutting away from
                # (measured 2026-08-31, workflow 17).
                "cut": p == 0 and bool(chunks),
                # The transition INTO this chunk, and whether the soundtrack
                # crosses it. `cut` above stays what it always was -- does this
                # chunk open a new shot -- because the runner reads it.
                "continuity": here_name if (p > 0 or chunks) else None,
                "carry_audio": audio_here == "carry" and bool(chunks),
                "shot": si if mode == "scene" else None,
                "part": (p + 1, n_parts) if n_parts > 1 else None,
                "both_clocks": on_both_clocks(run),
                "seed_mask": p > 0,      # only within a shot, never across a cut
            })

    off = sum(1 for c in chunks if not c["both_clocks"])
    if off and generated_audio:
        # the reassurance below is a V2V fact and would be a lie here: these are
        # the chunks the back-extension could not reach a shared run for
        notes.append(f"{off} chunk(s) are off the audio grid and the audio is "
                     f"GENERATED, so each one's rounding against the 40 Hz latent "
                     f"accumulates down the chain — no shared run was reachable "
                     f"for them without cutting or inventing content.")
    elif off:
        notes.append(f"{off} chunk(s) are off the audio grid. Source audio is "
                     f"PRESERVED and reassembled at exact sample boundaries, so "
                     f"this does not accumulate — unlike a generative chain.")

    info = {
        "chunks": len(chunks), "mode": mode, "shots": len(shots),
        "total_frames": total, "chunk_frames": size,
        "off_grid": off, "notes": notes,
        # APPLIED cuts only. These used to be whatever was passed in, so a
        # fixed plan reported cuts it had just thrown away.
        "cuts": sorted(set(int(c) for c in cuts)) if cuts else [],
        "cuts_ignored": ignored_cuts,
    }
    return chunks, info


def latent_frames(run):
    """Latent frames a legal run encodes to, under the (1,4,4,4,4) grouping.

    Same number as `timing.video_latent_t`. Kept here rather than imported
    because every torch-free module in this pack is import-free at the top level
    so the tests can load them as flat modules from any harness.
    """
    n = int(run)
    return 2 if n <= 5 else ((n - 5) // 17) * 5 + 2


def tokens_per_frame(render_w, render_h):
    """Video tokens in ONE latent frame at this canvas.

    The DiT patches the /16 latent 2x2, so a pixel canvas contributes
    `(w/32)*(h/32)` rows per latent frame. `budget.frame_rows` writes the same
    number from the patch side; this is the canvas side of it.
    """
    return (int(render_w) // 32) * (int(render_h) // 32)


def video_tokens(chunk_run, render_w, render_h):
    """Picture tokens a chunk of this length costs — the number cost goes as.

    Attention is roughly quadratic in the sequence length, so this is the figure
    that decides whether a chunk size is affordable, and it is invisible
    everywhere else in the graph.
    """
    return latent_frames(chunk_run) * tokens_per_frame(render_w, render_h)


def ref_share(chunk_run, ref_tokens, render_w, render_h):
    """A reference set's share of the picture tokens for a chunk of this length.

    The reason chunk length is an identity lever and not just a cost one.
    """
    total = video_tokens(chunk_run, render_w, render_h) + int(ref_tokens)
    return (int(ref_tokens) / total) if total else 0.0


def describe(chunks, info, render=None, ref_tokens=0, render_from=""):
    """The report the node prints. This is the deliverable, not a debug aid.

    With a render size it prints BOTH numbers that a chunk length decides: the
    picture tokens the chunk costs, and the share the references keep against
    them. Neither is judged — the crossover where a reference stops winning is
    unmeasured, and every quality proxy this project has built has failed, so
    these are reported as data and the eye decides.

    `render_from` says where the size came from, because a size derived from the
    source clip and a size typed into a widget are different claims and a report
    that does not distinguish them is guessing on the reader's behalf.
    """
    L = [f"chunk_mode: {info['mode']}    {info['chunks']} chunks from "
         f"{info['total_frames']} frames"]
    if info["mode"] == "scene":
        L[0] += f" across {info['shots']} shot(s)"
    if render:
        line = (f"render {render[0]}x{render[1]}"
                + (f" ({render_from})" if render_from else "")
                + f"    {tokens_per_frame(*render):,} tokens per latent frame")
        if ref_tokens:
            line += f", {int(ref_tokens):,} reference tokens"
        L.append(line)
    for i, c in enumerate(chunks, 1):
        tag = ""
        if c["shot"] is not None:
            tag = f"shot {c['shot'] + 1}"
            if c["part"]:
                tag += f" ({c['part'][0]} of {c['part'][1]})"
        over = c["keep_from"] - c["start"]
        pad = "" if not over else (
            f" (+{over}f overlap"
            f"{', ' + str(c['pin']) + 'f pinned' if c.get('pin') else ''}"
            f", dropped at the join)")
        clocks = "both clocks" if c["both_clocks"] else "OFF grid"
        seed = "" if c["seed_mask"] else "  [track restarts]"
        # fixed-width columns, because the point of the report is scanning DOWN
        # it for the chunk that is different from the others. The variable-length
        # prose -- the overlap note, the track-restart flag -- goes after them
        # for the same reason: padding it to its worst case would put 40 blank
        # columns on every line that does not have one.
        tok = "" if not render else \
            f"  {video_tokens(c['run'], *render):>9,} tokens"
        share = "" if not (render and ref_tokens) else \
            f"   refs {ref_share(c['run'], ref_tokens, *render) * 100:4.1f}%"
        L.append(f"  {i:02d}  frames {c['start']:5d}-{c['end']:5d}  {tag:<18}"
                 f"{c['length']:4d}f{tok}{share}   {clocks}{pad}{seed}")
    if info["cuts"]:
        shown = ", ".join(str(c) for c in info["cuts"][:16])
        more = "" if len(info["cuts"]) <= 16 else f" (+{len(info['cuts']) - 16} more)"
        L.append(f"  cuts APPLIED at: {shown}{more}")
    if info.get("cuts_ignored"):
        shown = ", ".join(str(c) for c in info["cuts_ignored"][:16])
        L.append(f"  cuts IGNORED (chunk_mode is not 'scene'): {shown}")
    for n in info["notes"]:
        L.append(f"  NOTE {n}")
    return "\n".join(L)
