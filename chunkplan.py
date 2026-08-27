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


def plan(total_frames, chunk_frames=90, mode="fixed", cuts=None, min_chunk=39):
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

    if total <= 0:
        return [], {"chunks": 0, "mode": mode, "notes": ["empty clip"]}

    if mode == "scene" and not cuts:
        notes.append("scene mode asked for but no cuts were found — fell back to "
                     "fixed. Hard cuts only; a dissolve will not trip the "
                     "threshold.")
        mode = "fixed"

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
        parts, pos = [], s0
        while s1 - pos > size + int(min_chunk):
            parts.append((pos, pos + size))
            pos += size
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
            keep_from = start
            if deficit and start - deficit >= s0:
                # room inside this SHOT — never reach back across a cut
                start -= deficit
                keep_from = start + deficit
            elif deficit:
                # nothing behind it: a shot shorter than one legal run. Trim to
                # the run below rather than generate frames that do not exist.
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
                "shot": si if mode == "scene" else None,
                "part": (p + 1, n_parts) if n_parts > 1 else None,
                "both_clocks": on_both_clocks(run),
                "seed_mask": p > 0,      # only within a shot, never across a cut
            })

    off = sum(1 for c in chunks if not c["both_clocks"])
    if off:
        notes.append(f"{off} chunk(s) are off the audio grid. Source audio is "
                     f"PRESERVED and reassembled at exact sample boundaries, so "
                     f"this does not accumulate — unlike a generative chain.")

    info = {
        "chunks": len(chunks), "mode": mode, "shots": len(shots),
        "total_frames": total, "chunk_frames": size,
        "off_grid": off, "notes": notes,
        "cuts": sorted(set(int(c) for c in cuts)) if cuts else [],
    }
    return chunks, info


def ref_share(chunk_run, ref_tokens, render_w, render_h):
    """A reference set's share of the picture tokens for a chunk of this length.

    The reason chunk length is an identity lever and not just a cost one.
    """
    lt = 2 if chunk_run <= 5 else ((int(chunk_run) - 5) // 17) * 5 + 2
    video = lt * (int(render_w) // 32) * (int(render_h) // 32)
    total = video + int(ref_tokens)
    return (int(ref_tokens) / total) if total else 0.0


def describe(chunks, info, render=None, ref_tokens=0):
    """The report the node prints. This is the deliverable, not a debug aid."""
    L = [f"chunk_mode: {info['mode']}    {info['chunks']} chunks from "
         f"{info['total_frames']} frames"]
    if info["mode"] == "scene":
        L[0] += f" across {info['shots']} shot(s)"
    for i, c in enumerate(chunks, 1):
        tag = ""
        if c["shot"] is not None:
            tag = f"shot {c['shot'] + 1}"
            if c["part"]:
                tag += f" ({c['part'][0]} of {c['part'][1]})"
        over = c["keep_from"] - c["start"]
        pad = "" if not over else f" (+{over}f overlap, dropped at the join)"
        clocks = "both clocks" if c["both_clocks"] else "OFF grid"
        seed = "" if c["seed_mask"] else "  [track restarts]"
        share = ""
        if render and ref_tokens:
            share = f"  refs {ref_share(c['run'], ref_tokens, *render) * 100:4.1f}%"
        L.append(f"  {i:02d}  frames {c['start']:5d}-{c['end']:5d}  {tag:<18}"
                 f"{c['length']:4d}f{pad:<18} {clocks}{share}{seed}")
    if info["cuts"]:
        shown = ", ".join(str(c) for c in info["cuts"][:16])
        more = "" if len(info["cuts"]) <= 16 else f" (+{len(info['cuts']) - 16} more)"
        L.append(f"  cuts detected at: {shown}{more}")
    for n in info["notes"]:
        L.append(f"  NOTE {n}")
    return "\n".join(L)
