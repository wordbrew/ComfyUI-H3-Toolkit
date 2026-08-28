"""Run a chunk plan over a long clip: H3 Chunk Open / H3 Chunk Close.

WHAT THIS IS
  A pair of nodes with YOUR chain wired between them. Open hands out one chunk's
  slices; Close captures the nodes you put in the middle, clones them once per
  chunk with that chunk's slices, and reassembles the result. One queue.

WHY A PAIR AND NOT ONE NODE
  A single node doing all of it has to swallow the clip, the mask, the model, the
  VAEs, the sampler, the references and the prompt, and hand back finished video.
  You cannot see what it did, you cannot change the sampler for one chunk, and a
  failure in chunk 7 is a log line rather than a red node. With a pair, the middle
  is an ordinary graph -- yours to read, edit and debug -- and every chunk
  inherits whatever you change there.

HOW THE CLONING WORKS
  `Close` reads DYNPROMPT (the live graph), walks back to its `Open`, and takes
  the INTERSECTION of "downstream of Open" and "upstream of Close" as the body.
  The intersection matters: a VAE loader feeding your chain from outside is
  upstream of Close but not downstream of Open, so it stays a single shared node
  instead of being cloned once per chunk. Verified on 2026-08-26 -- a probe with
  a VAELoader feeding the body kept it external and referenced it 8 times across
  4 clones rather than loading it 4 times.

  The chunk count comes from the plan, so all N copies are emitted in ONE
  expansion. No recursion, no termination condition.

WHAT IT DELIBERATELY DOES NOT DO
  No mask seeding across a cut. `H3ChunkPlan` marks each chunk `seed_mask`, and
  the first chunk of every shot is False: seeding SAM3 with the previous chunk's
  final mask across a scene change points it at where the subject stood in a
  DIFFERENT shot, which is worse than re-detecting. Mask instability frees
  appearance -- that is the hair-colour mechanism -- so a boundary that jumps
  shows up as the character changing at the seam.
"""

import logging

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on host build
    GraphBuilder = None
    is_link = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

CATEGORY = "MiniMax H3/long-form"
OPEN_CLASS = "H3ChunkOpen"


def _links(node):
    for key, val in (node.get("inputs") or {}).items():
        if is_link(val):
            yield key, val


def upstream_of(dynprompt, start_id, stop_class):
    """Nodes reachable walking BACK from start_id, not descending past stop_class."""
    seen, stack, opens = set(), [start_id], []
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        try:
            node = dynprompt.get_node(nid)
        except Exception:
            continue
        if node.get("class_type") == stop_class:
            opens.append(nid)
            continue
        for _, val in _links(node):
            stack.append(val[0])
    seen.discard(start_id)
    return seen, opens


def downstream_of(dynprompt, start_id):
    """Nodes reachable walking FORWARD from start_id."""
    reach, changed = set(), True
    ids = list(dynprompt.all_node_ids())
    while changed:
        changed = False
        for nid in ids:
            if nid in reach:
                continue
            try:
                node = dynprompt.get_node(nid)
            except Exception:
                continue
            for _, val in _links(node):
                if val[0] == start_id or val[0] in reach:
                    reach.add(nid)
                    changed = True
                    break
    return reach


def slice_chunk(plan, index, source_images, mask=None, source_audio=None,
                prev_images=None, context_frames=39, extra_images=None):
    """One chunk's slices, plus the handoff from the previous chunk.

    V2V: source_images is wired, every chunk is cut from footage that exists, and
    prev_images is not needed -- the source pins composition on both sides.

    FRESH GENERATION: there is no source. Continuity has to be carried, and this
    returns the two things wave 14 used to get a 57.6s take with no join above
    3 sigma:

      keyframe        the previous chunk's last DECODED FRAME. Pixel, never
                      latent. Wave 8a arm B fed the last latent frame into the
                      keyframe slot and error roughly DOUBLED per link: the video
                      VAE is temporal, so its final latent frame carries ~4 pixel
                      frames of motion plus causal conv state, while the keyframe
                      slot expects a single-image encode. Same shape, different
                      distribution, and the model compensates with contrast.

      context_frames  the previous chunk's TAIL. A still frame carries pose but
                      not velocity -- a keyframe-only chain eroded motion -19%
                      monotonically (conclusion 8), where context-fed links held
                      flat. This is what the keyframe cannot do.
    """
    import torch

    chunks = (plan or {}).get("chunks") or []
    kf = ctx = None
    if prev_images is not None and int(prev_images.shape[0]) > 0:
        kf = prev_images[-1:]
        k = max(1, min(int(context_frames), int(prev_images.shape[0])))
        ctx = prev_images[-k:]

        # V2V: the frames to carry are the ones this chunk's head OVERLAPS, and
        # that is not always the previous chunk's tail. A chunk that reached
        # backwards for a legal run starts EARLIER than the previous chunk
        # ended, so its head sits in the middle of what came before rather than
        # at the end of it. Taking the last N there pins frames from further
        # along the clip than the chunk actually begins, and the pin fights the
        # source instead of anchoring to it.
        #
        # `prev_images` is the previous chunk's output AFTER its own head was
        # trimmed, so it starts at that chunk's keep_from.
        i0 = max(0, min(int(index), len(chunks) - 1)) if chunks else 0
        pin = int(chunks[i0].get("pin", 0)) if chunks else 0
        if chunks and i0 > 0 and source_images is not None and pin > 0:
            base = int(chunks[i0 - 1].get("keep_from", chunks[i0 - 1]["start"]))
            off = max(0, int(chunks[i0]["start"]) - base)
            avail = int(prev_images.shape[0])
            if off + pin <= avail:
                ctx = prev_images[off:off + pin]
            else:
                ctx = prev_images[max(0, avail - pin):]
                logging.info("H3ChunkSlice: chunk %d wanted frames %d-%d of the "
                             "previous chunk but only %d exist; pinned the last "
                             "%d instead", i0, off, off + pin, avail, pin)

    if source_images is None:
        # fresh generation: nothing to cut, the chunk is a schedule entry
        i = max(0, min(int(index), max(0, len(chunks) - 1)))
        c = chunks[i] if chunks else {"start": 0, "end": 0, "run": 0,
                                      "both_clocks": True, "seed_mask": False}
        text = (f"chunk {i + 1} of {len(chunks) or 1}: generating {c['run']} frames"
                f"{'' if c['both_clocks'] else '  OFF audio grid'}"
                f"{'' if kf is None else '  <- keyframe + context carried'}")
        return (None, None, source_audio, c["run"], i,
                {"plan": plan, "index": i}, text, len(chunks) or 1, kf, ctx, None,
                int(c.get("pin", 0)))

    if not chunks:
        n = int(source_images.shape[0])
        return (source_images, mask, source_audio, n, 0,
                {"plan": plan, "index": 0}, "empty plan — clip passed through", 1,
                kf, ctx, extra_images, 0)

    i = max(0, min(int(index), len(chunks) - 1))
    c = chunks[i]
    s0, e0 = int(c["start"]), int(c["end"])
    imgs = source_images[s0:e0]
    msk = mask[s0:e0] if mask is not None else torch.zeros(
        (e0 - s0, int(source_images.shape[1]), int(source_images.shape[2])),
        device=source_images.device)

    aud = source_audio
    if source_audio is not None:
        wf = source_audio.get("waveform")
        sr = int(source_audio.get("sample_rate", 48000))
        if wf is not None:
            # exact integer sample boundaries -- never resample or crossfade,
            # which smears the beat (wave 15)
            aud = {"waveform": wf[..., int(round(s0 / 24.0 * sr)):
                                       int(round(e0 / 24.0 * sr))],
                   "sample_rate": sr}

    text = (f"chunk {i + 1} of {len(chunks)}: frames {s0}-{e0} ({e0 - s0}f, run "
            f"{c['run']}){'' if c['both_clocks'] else '  OFF audio grid'}"
            f"{'' if c['seed_mask'] else '  [track restarts]'}")
    xtra = extra_images[s0:e0] if extra_images is not None else None
    return (imgs, msk, aud, e0 - s0, i, {"plan": plan, "index": i}, text,
            len(chunks), kf, ctx, xtra, int(c.get("pin", 0)))


class H3ChunkOpen:
    """Start of the repeated section. Hands out one chunk's slices.

    On its own it emits chunk 0, so the graph renders as an ordinary single-chunk
    workflow and you can build and debug it without any cloning in the way. Close
    is what turns it into N.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_CHUNK_PLAN", {"tooltip": "From H3 Chunk Plan."}),
        }, "optional": {
            "source_images": ("IMAGE", {"tooltip": "The WHOLE clip, for V2V. "
                                        "Leave UNWIRED for fresh generation — "
                                        "then `keyframe` and `context` carry "
                                        "continuity instead."}),
            "extra_images": ("IMAGE", {"tooltip": "A derived stream cut with the "
                             "images — depth, pose, an edge map. Compute it on "
                             "the WHOLE clip and slice it here: depth normalises "
                             "per image, so computing it per chunk would shift "
                             "the scene's depth range between chunks."}),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 360,
                               "step": 17,
                               "tooltip": "Fresh generation: how much of the "
                                          "previous chunk's tail to hand forward. "
                                          "39 lands exactly on the audio grid; 22 "
                                          "was used in wave 14 and is the worst "
                                          "legal value (36.667 audio steps)."}),
            "mask": ("MASK", {"tooltip": "Full-length mask, sliced with the "
                                         "images. Leave unwired to build the "
                                         "mask INSIDE the body instead, which is "
                                         "what lets SAM3 track per chunk."}),
            "source_audio": ("AUDIO",),
        }}

    # chunk_count appended LAST -- saved workflows store slot indices, so a new
    # output goes on the end or every existing link silently shifts
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "INT", "H3_CHUNK_FLOW",
                    "STRING", "INT", "IMAGE", "IMAGE", "IMAGE", "INT", "LATENT")
    RETURN_NAMES = ("images", "mask", "audio", "length", "chunk_index",
                    "flow", "info", "chunk_count", "keyframe", "context",
                    "extra", "pin", "prev_latent")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Hands out one chunk of a long clip. Wire your chain from here "
                   "to H3 Chunk Close, which repeats it for every chunk.")

    def go(self, plan, source_images=None, mask=None, source_audio=None,
           extra_images=None, context_frames=39):
        out = slice_chunk(plan, 0, source_images, mask, source_audio,
                          context_frames=context_frames,
                          extra_images=extra_images)
        n = len(((plan or {}).get("chunks")) or [])
        note = (f"\n  wire your chain from here into H3 Chunk Close; it repeats "
                f"this for all {n} chunks.") if n > 1 else ""
        # prev_latent is None on Open by definition: it stands in for chunk 0,
        # and chunk 0 has nothing before it. Close fills it in for the clones.
        return out[:6] + (out[6] + note,) + out[7:] + (None,)


class H3ChunkClose:
    """Capture the chain wired above and run it once per chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("H3_CHUNK_FLOW", {"tooltip": "From H3 Chunk Open."}),
                # LAZY. This node needs the LINK, to know where your chain ends;
                # it never needs the value. Evaluated eagerly, ComfyUI rendered
                # the un-cloned body once to satisfy it and threw the result
                # away, THEN expanded -- 4 sampler passes for 3 chunks, measured
                # 2026-08-28. check_lazy_status below asks for it only on the
                # paths that actually pass it through.
                "images": ("IMAGE", {"lazy": True,
                                     "tooltip": "The END of your chain — the "
                                                "processed frames for one chunk."}),
            },
            "optional": {
                "audio": ("AUDIO", {"lazy": True,
                          "tooltip": "Optional, and only for its LINK: the "
                                     "per-chunk VAE Decode Audio. Wire it and "
                                     "the joined audio comes out of `audio`, cut "
                                     "on the SAME frame boundaries as the "
                                     "picture. Leave it unwired and the run is "
                                     "silent — the model still generates audio "
                                     "into the latent, it is just discarded at "
                                     "the decode."}),
                "latent": ("LATENT", {"lazy": True,
                           "tooltip": "Optional, and also only for its LINK: the "
                                      "sampler output. Wire it to chain chunks in "
                                      "LATENT space — each chunk's latent reaches "
                                      "the next one's `prev_latent` with no "
                                      "decode/re-encode round trip."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    # `audio` is APPENDED — saved graphs store slot indices by position
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("images", "info", "audio")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Repeat the chain between H3 Chunk Open and here for every "
                   "chunk in the plan, then join the results.")

    def check_lazy_status(self, flow, images=None, audio=None, latent=None,
                          dynprompt=None, unique_id=None):
        """Which inputs actually have to be computed.

        Returning [] on the expansion path is the whole point: the body is about
        to be cloned once per chunk, so evaluating the original copy first is a
        whole wasted render. Every path that falls back to passing `images`
        through asks for it, so a misconfigured graph still returns something.
        """
        if GraphBuilder is None or dynprompt is None or unique_id is None:
            return ["images"]
        chunks = ((flow or {}).get("plan") or {}).get("chunks") or []
        if len(chunks) <= 1:
            return ["images"]          # nothing to repeat; the one result IS it
        try:
            _, opens = upstream_of(dynprompt, unique_id, OPEN_CLASS)
        except Exception:
            return ["images"]
        return [] if opens else ["images"]

    def go(self, flow, images=None, audio=None, latent=None, dynprompt=None,
           unique_id=None):
        def _bail(msg):
            # `images` is lazy, so on a path check_lazy_status did not predict it
            # can be None -- and handing None down the graph as an IMAGE fails
            # somewhere far away. Fail here, where the cause is.
            if images is None:
                raise ValueError(f"H3 Chunk Close: {msg}")
            return (images, msg, audio)

        if GraphBuilder is None:
            return _bail(f"GraphBuilder unavailable: {_IMPORT_ERROR}")
        if dynprompt is None or unique_id is None:
            return _bail("DYNPROMPT / UNIQUE_ID were not supplied by this "
                         "ComfyUI build, so the chain cannot be captured.")

        plan = (flow or {}).get("plan") or {}
        chunks = plan.get("chunks") or []
        if len(chunks) <= 1:
            return _bail("plan has one chunk or none — nothing to repeat, "
                         "passing the single result through.")

        up, opens = upstream_of(dynprompt, unique_id, OPEN_CLASS)
        if not opens:
            return _bail("No H3 Chunk Open found upstream. Wire `flow` from one.")
        open_id = opens[0]
        down = downstream_of(dynprompt, open_id)

        # THE INTERSECTION. Upstream-of-Close alone drags in every loader feeding
        # the chain from outside and would clone the VAE once per chunk.
        body_ids = sorted(up & down)
        if not body_ids:
            return _bail("Nothing between Open and Close — wire your chain "
                         "through the middle.")
        body = {nid: dynprompt.get_node(nid) for nid in body_ids}
        external = sorted(up - down)

        close_inputs = dynprompt.get_node(unique_id).get("inputs") or {}
        tail_link = close_inputs.get("images")
        # the sampler, if the graph wants to chain in latent space
        audio_link = close_inputs.get("audio")
        latent_link = close_inputs.get("latent")
        if is_link(latent_link) and latent_link[0] not in \
                {n for n in (upstream_of(dynprompt, unique_id, OPEN_CLASS)[0])}:
            latent_link = None
        if not is_link(tail_link):
            return _bail("`images` is not connected to anything cloneable.")
        if tail_link[0] not in body:
            return _bail(f"`images` comes from node {tail_link[0]}, which is "
                         f"not inside the Open..Close section. Wire the END of "
                         f"your chain into `images`.")

        graph = GraphBuilder()
        outs, lat_outs, aud_outs = [], [], []
        for ci, c in enumerate(chunks):
            # feed the slicer from whatever fed Open -- the video loader is a
            # shared external node, referenced not cloned
            open_node = dynprompt.get_node(open_id)
            open_inputs = open_node.get("inputs") or {}
            open_widgets = {k: v for k, v in open_inputs.items()
                            if not is_link(v)}
            src_kw = {"plan": plan, "chunk_index": ci}
            for k in ("source_images", "mask", "source_audio", "extra_images"):
                v = open_inputs.get(k)
                if is_link(v):
                    src_kw[k] = [v[0], v[1]]
            # FRESH GENERATION makes the chain sequential: chunk N's keyframe
            # and motion context are chunk N-1's DECODED output, so clone N
            # depends on clone N-1. In V2V this link is simply unused, because
            # the source pins both sides of every seam.
            if outs:
                src_kw["prev_images"] = outs[-1]
            if lat_outs:
                # LATENT chaining: chunk N's context prefix is copied straight
                # from chunk N-1's sampler output, so nothing is decoded and
                # re-encoded between links. The re-encode is what climbed
                # contrast down a chain.
                src_kw["prev_latent"] = lat_outs[-1]
            src_kw["context_frames"] = int(open_widgets.get("context_frames", 39))
            src = graph.node("H3ChunkSlice", id=f"slice{ci}", **src_kw)
            mapping = {nid: graph.node(body[nid]["class_type"], id=f"c{ci}_{nid}")
                       for nid in body_ids}
            for nid in body_ids:
                new = mapping[nid]
                for key, val in (dynprompt.get_node(nid).get("inputs") or {}).items():
                    if not is_link(val):
                        new.set_input(key, val)
                    elif val[0] == open_id:
                        # remap Open's outputs onto this chunk's slicer, which
                        # carries the same output order
                        new.set_input(key, src.out(val[1]))
                    elif val[0] in mapping:
                        new.set_input(key, mapping[val[0]].out(val[1]))
                    else:
                        new.set_input(key, [val[0], val[1]])   # shared, not cloned
            tail = mapping[tail_link[0]].out(tail_link[1])
            # A chunk extended backwards to reach a legal run regenerates frames
            # the previous chunk already covered. Drop them here rather than at
            # the plan, so the model still sees the run it needs and the joined
            # clip keeps every source frame exactly once.
            skip = int(c.get("keep_from", c["start"])) - int(c["start"])
            if skip > 0:
                tail = graph.node("ImageFromBatch", id=f"trim{ci}", image=tail,
                                  batch_index=skip,
                                  length=int(c["end"]) - int(c["keep_from"])).out(0)
            outs.append(tail)

            # AUDIO gets the same treatment as the picture and on the same
            # boundaries: a chunk regenerates the span its pin held, so both
            # streams drop it, or the sound walks away from the picture a little
            # further at every seam. Sliced, never resampled or crossfaded --
            # wave 15 measured that smearing the beat.
            if is_link(audio_link) and audio_link[0] in mapping:
                a = mapping[audio_link[0]].out(audio_link[1])
                if skip > 0:
                    a = graph.node("H3AudioTrimFrames", id=f"atrim{ci}", audio=a,
                                   skip_frames=skip,
                                   keep_frames=int(c["end"]) - int(c["keep_from"])
                                   ).out(0)
                aud_outs.append(a)
            if is_link(latent_link) and latent_link[0] in mapping:
                lat_outs.append(mapping[latent_link[0]].out(latent_link[1]))

        joined = outs[0]
        for nxt in outs[1:]:
            joined = graph.node("ImageBatch", image1=joined, image2=nxt).out(0)

        # A fresh-generation tail grows FORWARD to reach a legal run its pin can
        # cover, so the last chunk can end past what was asked for -- 300 frames
        # requested comes back as 345. Trim to the ask, so the number typed into
        # the plan is the number delivered. V2V never overshoots (there is no
        # footage past the end), so this is a no-op there.
        joined_audio = None
        if aud_outs:
            joined_audio = aud_outs[0]
            for nxt in aud_outs[1:]:
                joined_audio = graph.node("AudioConcat", audio1=joined_audio,
                                          audio2=nxt, direction="after").out(0)

        want = int(plan.get("total_frames") or 0)
        made = sum(int(c["end"]) - int(c.get("keep_from", c["start"]))
                   for c in chunks)
        if want and made > want:
            joined = graph.node("ImageFromBatch", id="fit", image=joined,
                                batch_index=0, length=want).out(0)
            if joined_audio is not None:
                joined_audio = graph.node("H3AudioTrimFrames", id="afit",
                                          audio=joined_audio, skip_frames=0,
                                          keep_frames=want).out(0)

        report = [
            f"H3 CHUNK CLOSE — {len(chunks)} chunks",
            ("  chaining in LATENT space — no decode/re-encode between chunks"
             if lat_outs else
             "  `latent` not wired — chunks carry only what you route through "
             "Open's own outputs"),
            f"  body ({len(body_ids)} node(s), cloned per chunk):",
        ]
        for nid in body_ids:
            report.append(f"      {nid}  {body[nid]['class_type']}")
        report.append(f"  shared, NOT cloned: {len(external)}")
        for nid in external[:10]:
            try:
                report.append(f"      {nid}  {dynprompt.get_node(nid)['class_type']}")
            except Exception:
                pass
        report.append("  audio joined on the same frame boundaries as the picture"
                      if aud_outs else
                      "  `audio` not wired — the model generates audio into the "
                      "latent and the decode throws it away")
        if want and made > want:
            report.append(f"  generated {made} frames for the {want} asked — the "
                          f"tail grew forward so its pin could reach the cut; "
                          f"the extra is trimmed off the end")
        expanded = graph.finalize()
        overlap = sum(int(c.get("keep_from", c["start"])) - int(c["start"])
                      for c in chunks)
        if overlap:
            report.append(f"  {overlap} overlap frame(s) regenerated and dropped "
                          f"at the join, so a short tail still gets a legal run")
        report.append(f"  emitted {len(expanded)} nodes across {len(chunks)} chunks")
        logging.info("H3ChunkClose: %s", report[0])
        return {"expand": expanded,
                "result": (joined, "\n".join(report), joined_audio)}


class H3ChunkSlice:
    """Internal. One chunk's slices, emitted by Close — not for the menu.

    Mirrors H3ChunkOpen's output order exactly, because Close remaps links from
    Open onto this by output INDEX.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_CHUNK_PLAN",),
            "chunk_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
        }, "optional": {
            "source_images": ("IMAGE",),
            "mask": ("MASK",),
            "source_audio": ("AUDIO",),
            "prev_images": ("IMAGE",),
            "extra_images": ("IMAGE",),
            "context_frames": ("INT", {"default": 39}),
            "prev_latent": ("LATENT",),
        }}

    # chunk_count appended LAST -- saved workflows store slot indices, so a new
    # output goes on the end or every existing link silently shifts
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "INT", "H3_CHUNK_FLOW",
                    "STRING", "INT", "IMAGE", "IMAGE", "IMAGE", "INT", "LATENT")
    RETURN_NAMES = ("images", "mask", "audio", "length", "chunk_index",
                    "flow", "info", "chunk_count", "keyframe", "context",
                    "extra", "pin", "prev_latent")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DEPRECATED = True          # keeps it out of the node menu
    DESCRIPTION = "Internal to H3 Chunk Close."

    def go(self, plan, chunk_index, source_images=None, mask=None,
           source_audio=None, prev_images=None, extra_images=None,
           context_frames=39, prev_latent=None):
        return slice_chunk(plan, chunk_index, source_images, mask, source_audio,
                           prev_images, context_frames,
                           extra_images) + (prev_latent,)


class H3ChunkContext:
    """Start a chunk on the previous chunk's finished frames, in PIXELS.

    WHAT IT DOES
      Overwrites the first `pin` frames of this chunk's source with the tail of
      the previous chunk's output, and blanks the mask over them so nothing
      regenerates there. Everything downstream then runs unchanged: one encode,
      one mask, on a clip whose opening already IS the swapped character.

    WHY IT SITS HERE AND NOT AFTER THE CONDITIONING
      The obvious wiring -- a context node and H3 Mask Inpaint in series -- does
      not work. Both write `samples` and `noise_mask` wholesale, so the second
      discards the first. Worse, H3 Mask Inpaint builds its latent as
      `vae.encode(source_images)` and never reads the incoming samples, so
      downstream it would overwrite the carried frames with raw source footage.
      Doing it in pixels, before anything encodes, means there is only ever one
      mask and one encode.

    WHY PIXELS AND NOT A LATENT SLICE
      H3LatentPin in this pack does the latent version and carries a warning
      from nine rounds of measurement: a visible cut where the pin ends. The
      VAE's FRAME_PER_TOKEN grouping is POSITIONAL -- latent steps built for
      positions 45..51 do not mean the same thing at 0..6. Re-encoding pixels at
      the front is why this path works and that one does not.

    HOW LONG
      Only 39, 22, 5 and 1 pixel frames encode to distinct VAE runs, so `pin`
      comes from the plan already snapped. 1 frame is a still keyframe: it fixes
      position and identity, but a keyframe-only chain eroded motion 19%
      monotonically down the links, because a still carries pose and not
      velocity. 22 held flat and is the usual answer. 39 costs more overlap for
      no measured gain over 22 in a V2V pass, where the source pins motion too.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "This chunk's source frames, from "
                                            "H3 Chunk Open."}),
            "mask": ("MASK", {"tooltip": "This chunk's subject mask."}),
            "pin": ("INT", {"default": 22, "min": 0, "max": 39,
                            "tooltip": "Frames to carry. From H3 Chunk Open's "
                                       "`pin` output, which is 0 on the first "
                                       "chunk of every shot."}),
        }, "optional": {
            "context_images": ("IMAGE", {"tooltip": "The PREVIOUS chunk's "
                                         "finished frames. Unwired, or on the "
                                         "first chunk, this node passes "
                                         "everything through untouched."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "mask", "pinned", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Overwrite a chunk's opening frames with the previous chunk's "
                   "output and blank the mask there, so the take continues "
                   "instead of restarting.")

    def go(self, images, mask, pin, context_images=None):
        import torch

        n = int(images.shape[0])
        if context_images is None or pin <= 0 or int(context_images.shape[0]) == 0:
            why = ("first chunk of the shot — nothing behind it to carry"
                   if pin <= 0 else "no context wired")
            return (images, mask, 0, f"H3 CHUNK CONTEXT: passthrough ({why})")

        k = min(int(pin), int(context_images.shape[0]), n - 1)
        if k != int(pin):
            logging.info("H3ChunkContext: pin %d -> %d (limited by what was "
                         "available)", int(pin), k)
        if k <= 0:
            return (images, mask, 0, "H3 CHUNK CONTEXT: passthrough (chunk too "
                                     "short to carry anything)")

        tail = context_images[-k:]
        if tail.shape[1:] != images.shape[1:]:
            # The crop box moved between chunks. Resizing would hide it and the
            # pin would fight the generation instead of anchoring it, so say
            # what actually happened.
            raise ValueError(
                f"the previous chunk is {int(tail.shape[2])}x{int(tail.shape[1])} "
                f"but this one is {int(images.shape[2])}x{int(images.shape[1])}. "
                f"The pinned frames have to line up pixel for pixel with what "
                f"follows them, so every chunk must share ONE crop box — move "
                f"H3 Subject Crop outside the Open..Close section, or set it to "
                f"`static` on the whole clip.")

        out = images.clone()
        out[:k] = tail.to(out.device, out.dtype)
        msk = mask.clone()
        msk[:k] = 0.0                       # 0 = preserve, so nothing regenerates

        held = float(mask[:k].mean()) * 100
        info = (f"H3 CHUNK CONTEXT: {k} frame(s) carried from the previous chunk "
                f"and held\n  the mask over them covered {held:.1f}% — now 0, so "
                f"the opening is\n  reproduced rather than regenerated, and the "
                f"join drops it")
        logging.info("H3ChunkContext: %d frame(s) pinned", k)
        return {"ui": {"h3char": [info]}, "result": (out, msk, k, info)}


class H3ChunkLatentContext:
    """Copy the previous chunk's latent tail into this chunk's prefix.

    WHY THIS WRAPPER EXISTS AND DOES NOT REIMPLEMENT ANYTHING
      `MiniMaxH3GeneratedAVMaskedContext` (ComfyUI-H3-Motion-Context-MultiRef)
      already does the copy correctly -- exact AV prefix runs, per-stream mask 0,
      the half-cosine audio release -- and it is what that pack's own six-link
      example uses for links 2..6. What it cannot do is sit in a body that is
      cloned per chunk, because `source_latent` is REQUIRED and chunk 0 has no
      previous chunk. So this delegates to it when there is a prefix to copy and
      passes the latent straight through when there is not.

    WHY LATENT HERE AND PIXELS IN H3ChunkContext
      They are different jobs. A V2V swap already has a mask writer -- H3 Mask
      Inpaint owns `samples` and `noise_mask` and rebuilds the latent from
      `vae.encode(source_images)` -- so a second latent writer is discarded, and
      the carry has to happen in pixels before anything encodes. Generating from
      nothing, no one else owns the latent, and going through pixels would mean
      a decode and re-encode per link, which is the round trip that climbed
      contrast down a chain.

    THE PREFIX IS REGENERATED, NOT FREE
      `context_length` frames of every chunk reproduce what the last one already
      delivered. Wire `pin` from H3 Chunk Open: the plan sizes the overlap, marks
      `keep_from` past it, and H3 Chunk Close drops it at the join, so the prefix
      costs compute and never appears twice. Set the plan's `context` to 39 --
      that node floors at 39 because it protects audio too, and 39 is the
      smallest count landing on both the 24 fps and 40 Hz grids.
    """

    CONTEXT_NODE = "MiniMaxH3GeneratedAVMaskedContext"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "This chunk's fresh target latent, "
                                             "from the H3 conditioning node."}),
        }, "optional": {
            "source_latent": ("LATENT", {"tooltip": "The PREVIOUS chunk's sampler "
                              "output, from H3 Chunk Open's `prev_latent`. Empty "
                              "on the first chunk, which is why this node exists "
                              "rather than the upstream one."}),
            "context_length": ("INT", {"default": 39, "min": 0, "max": 4096,
                               "tooltip": "Wire H3 Chunk Open's `pin`, so the "
                                          "number copied is the number the plan "
                                          "overlapped and the join drops. 0 "
                                          "passes through."}),
            "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 256,
                                    "tooltip": "Half-cosine release across the "
                                               "final audio-latent ticks of the "
                                               "prefix. 8 = 0.2s at 40 Hz."}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "trim_frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Chain generated chunks in latent space: the previous chunk's "
                   "tail becomes this one's protected prefix, with no decode and "
                   "re-encode in between.")

    def go(self, latent, source_latent=None, context_length=39,
           audio_feather_ticks=8):
        if source_latent is None or int(context_length) <= 0:
            why = ("first chunk — nothing before it to continue from"
                   if source_latent is None else "context_length 0")
            return (latent, 0, f"H3 CHUNK LATENT CONTEXT: passthrough ({why})")

        # resolve by NODE ID through the registry: the class name and the id
        # differ in these packs, and the id is what stays stable
        cls = None
        try:
            import nodes as comfy_nodes
            cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(
                self.CONTEXT_NODE)
        except Exception:
            pass
        if cls is None:
            raise ValueError(
                f"{self.CONTEXT_NODE} is not installed. Latent chunk chaining "
                f"needs ComfyUI-H3-Motion-Context-MultiRef — install it, or "
                f"leave `source_latent` unwired to run the chunks independently.")

        out = cls().prepare(latent=latent, source_latent=source_latent,
                            context_length=int(context_length),
                            audio_feather_ticks=int(audio_feather_ticks))
        new_latent, trim = (out[0], out[1]) if isinstance(out, (list, tuple)) \
            else (out, int(context_length))
        info = (f"H3 CHUNK LATENT CONTEXT: {trim} frame(s) copied from the "
                f"previous chunk and masked to 0\n  the plan overlapped by this "
                f"much, so H3 Chunk Close drops it at the join")
        logging.info("H3ChunkLatentContext: prefix %s frames", trim)
        return {"ui": {"h3char": [info]}, "result": (new_latent, int(trim), info)}


NODE_CLASS_MAPPINGS = {"H3ChunkOpen": H3ChunkOpen, "H3ChunkClose": H3ChunkClose,
                       "H3ChunkSlice": H3ChunkSlice,
                       "H3ChunkContext": H3ChunkContext,
                       "H3ChunkLatentContext": H3ChunkLatentContext}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkOpen": "H3 Chunk Open",
                              "H3ChunkClose": "H3 Chunk Close",
                              "H3ChunkSlice": "H3 Chunk Slice (internal)",
                              "H3ChunkContext": "H3 Chunk Context (carry the seam)",
                              "H3ChunkLatentContext":
                                  "H3 Chunk Latent Context (chain generated chunks)"}
