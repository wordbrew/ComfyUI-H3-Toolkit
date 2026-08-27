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
                prev_images=None, context_frames=39):
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

    if source_images is None:
        # fresh generation: nothing to cut, the chunk is a schedule entry
        i = max(0, min(int(index), max(0, len(chunks) - 1)))
        c = chunks[i] if chunks else {"start": 0, "end": 0, "run": 0,
                                      "both_clocks": True, "seed_mask": False}
        text = (f"chunk {i + 1} of {len(chunks) or 1}: generating {c['run']} frames"
                f"{'' if c['both_clocks'] else '  OFF audio grid'}"
                f"{'' if kf is None else '  <- keyframe + context carried'}")
        return (None, None, source_audio, c["run"], i,
                {"plan": plan, "index": i}, text, len(chunks) or 1, kf, ctx)

    if not chunks:
        n = int(source_images.shape[0])
        return (source_images, mask, source_audio, n, 0,
                {"plan": plan, "index": 0}, "empty plan — clip passed through", 1,
                kf, ctx)

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
    return (imgs, msk, aud, e0 - s0, i, {"plan": plan, "index": i}, text,
            len(chunks), kf, ctx)


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
                    "STRING", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "mask", "audio", "length", "chunk_index",
                    "flow", "info", "chunk_count", "keyframe", "context")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Hands out one chunk of a long clip. Wire your chain from here "
                   "to H3 Chunk Close, which repeats it for every chunk.")

    def go(self, plan, source_images=None, mask=None, source_audio=None,
           context_frames=39):
        out = slice_chunk(plan, 0, source_images, mask, source_audio,
                          context_frames=context_frames)
        n = len(((plan or {}).get("chunks")) or [])
        note = (f"\n  wire your chain from here into H3 Chunk Close; it repeats "
                f"this for all {n} chunks.") if n > 1 else ""
        return out[:6] + (out[6] + note,) + out[7:]


class H3ChunkClose:
    """Capture the chain wired above and run it once per chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("H3_CHUNK_FLOW", {"tooltip": "From H3 Chunk Open."}),
                "images": ("IMAGE", {"tooltip": "The END of your chain — the "
                                                "processed frames for one chunk."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Repeat the chain between H3 Chunk Open and here for every "
                   "chunk in the plan, then join the results.")

    def go(self, flow, images, dynprompt=None, unique_id=None):
        if GraphBuilder is None:
            return (images, f"GraphBuilder unavailable: {_IMPORT_ERROR}")
        if dynprompt is None or unique_id is None:
            return (images, "DYNPROMPT / UNIQUE_ID were not supplied by this "
                            "ComfyUI build, so the chain cannot be captured.")

        plan = (flow or {}).get("plan") or {}
        chunks = plan.get("chunks") or []
        if len(chunks) <= 1:
            return (images, "plan has one chunk or none — nothing to repeat, "
                            "passing the single result through.")

        up, opens = upstream_of(dynprompt, unique_id, OPEN_CLASS)
        if not opens:
            return (images, "No H3 Chunk Open found upstream. Wire `flow` from one.")
        open_id = opens[0]
        down = downstream_of(dynprompt, open_id)

        # THE INTERSECTION. Upstream-of-Close alone drags in every loader feeding
        # the chain from outside and would clone the VAE once per chunk.
        body_ids = sorted(up & down)
        if not body_ids:
            return (images, "Nothing between Open and Close — wire your chain "
                            "through the middle.")
        body = {nid: dynprompt.get_node(nid) for nid in body_ids}
        external = sorted(up - down)

        close_inputs = dynprompt.get_node(unique_id).get("inputs") or {}
        tail_link = close_inputs.get("images")
        if not is_link(tail_link):
            return (images, "`images` is not connected to anything cloneable.")
        if tail_link[0] not in body:
            return (images, f"`images` comes from node {tail_link[0]}, which is "
                            f"not inside the Open..Close section. Wire the END of "
                            f"your chain into `images`.")

        graph = GraphBuilder()
        outs = []
        for ci, c in enumerate(chunks):
            # feed the slicer from whatever fed Open -- the video loader is a
            # shared external node, referenced not cloned
            open_node = dynprompt.get_node(open_id)
            open_inputs = open_node.get("inputs") or {}
            open_widgets = {k: v for k, v in open_inputs.items()
                            if not is_link(v)}
            src_kw = {"plan": plan, "chunk_index": ci}
            for k in ("source_images", "mask", "source_audio"):
                v = open_inputs.get(k)
                if is_link(v):
                    src_kw[k] = [v[0], v[1]]
            # FRESH GENERATION makes the chain sequential: chunk N's keyframe
            # and motion context are chunk N-1's DECODED output, so clone N
            # depends on clone N-1. In V2V this link is simply unused, because
            # the source pins both sides of every seam.
            if outs:
                src_kw["prev_images"] = outs[-1]
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
            outs.append(mapping[tail_link[0]].out(tail_link[1]))

        joined = outs[0]
        for nxt in outs[1:]:
            joined = graph.node("ImageBatch", image1=joined, image2=nxt).out(0)

        report = [
            f"H3 CHUNK CLOSE — {len(chunks)} chunks",
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
        report.append(f"  emitted {len(body_ids) * len(chunks) + len(chunks) + max(0, len(chunks) - 1)} "
                      f"nodes across {len(chunks)} chunks")
        logging.info("H3ChunkClose: %s", report[0])
        return {"expand": graph.finalize(),
                "result": (joined, "\n".join(report))}


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
            "context_frames": ("INT", {"default": 39}),
        }}

    # chunk_count appended LAST -- saved workflows store slot indices, so a new
    # output goes on the end or every existing link silently shifts
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "INT", "H3_CHUNK_FLOW",
                    "STRING", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "mask", "audio", "length", "chunk_index",
                    "flow", "info", "chunk_count", "keyframe", "context")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DEPRECATED = True          # keeps it out of the node menu
    DESCRIPTION = "Internal to H3 Chunk Close."

    def go(self, plan, chunk_index, source_images=None, mask=None,
           source_audio=None, prev_images=None, context_frames=39):
        return slice_chunk(plan, chunk_index, source_images, mask, source_audio,
                           prev_images, context_frames)


NODE_CLASS_MAPPINGS = {"H3ChunkOpen": H3ChunkOpen, "H3ChunkClose": H3ChunkClose,
                       "H3ChunkSlice": H3ChunkSlice}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkOpen": "H3 Chunk Open",
                              "H3ChunkClose": "H3 Chunk Close",
                              "H3ChunkSlice": "H3 Chunk Slice (internal)"}
