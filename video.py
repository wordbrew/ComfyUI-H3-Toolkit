"""Conditioning and keyframes for MiniMax-H3.

WHAT UPSTREAM CANNOT DO
  MiniMaxH3ImageToVideo sets `minimax_keyframes` and MiniMaxH3ReferenceToVideo sets
  `minimax_refs`, on two different checkpoints, and nothing combines them. So you
  can have identity anchors OR a "start from this frame" signal, never both — which
  is exactly what chaining clips into a long take needs.

  Passing both is not merely unsupported, it is silently WRONG: PackedLayout places
  keyframes at cond_t = text_len (frame 0) and text_len + span - FRAME_RESCALE (last
  frame), which is where the target video sits ONLY when there are no refs. The refs
  branch then resets cursor = text_len and advances one slot per reference, moving
  the target video but leaving the keyframes behind, on top of the ref images.

  `patch_packed_layout()` below fixes that: references are laid down first, then
  keyframes are placed against the SAME cursor the target video uses. It also lifts
  the first/last-only restriction, so a keyframe can sit anywhere on the timeline.

UNTESTED — the arbitrary-position keyframe has never been run. First/last are
proven (they are what upstream does); a midpoint keyframe is a reasonable
extrapolation of the same positional maths, not a measured result.

  - A KEYFRAME is bound to frame 0 positionally, so it cannot act as a destination.
    That is why it is the continuation mechanism that works, where a reference
    video acts as a destination and the clip converges onto it.
"""

import inspect

import torch

import comfy.model_management
import node_helpers
from comfy_api.latest import io

from .avlatent import av
from .timing import snap_run, video_latent_t

CATEGORY = "MiniMax H3/video"

# Mirrors stock MiniMaxH3ReferenceToVideo's autogrow ceiling. Nine matters: a
# reference's share of the picture tokens falls as the clip lengthens, and nine
# large references on `max` is what holds a 294-frame render above the ratio a
# 39-frame one gets from three. See H3RefBudget.
MAX_REF_IMAGES = 9

# Core's own constant, imported so a change upstream follows rather than drifting.
try:
    from comfy_extras.nodes_minimax_h3 import REF_IMAGE_SHORT_EDGE
except Exception:  # pragma: no cover - core moved or renamed
    REF_IMAGE_SHORT_EDGE = 2048


#  layout patch: let keyframes and references coexist
# --------------------------------------------------------------------------- #

_PATCHED = False


def patch_packed_layout():
    """Replace PackedLayout so refs + keyframes can share one packed sequence.

    Idempotent, and a no-op if the H3 model module is not importable.
    """
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import comfy.ldm.minimax.model as M
    except Exception:
        return False

    Base = M.PackedLayout
    _BASE_PARAMS = set(inspect.signature(Base.__init__).parameters)
    if getattr(Base, "_h3_longform_patched", False):
        _PATCHED = True
        return True

    _frame_grid = M._frame_grid
    _video_grid = M._video_grid
    _audio_grid = M._audio_grid
    _video_t_spans = M._video_t_spans
    FRAME_RESCALE = M.FRAME_RESCALE

    class LongFormLayout(Base):
        _h3_longform_patched = True

        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            # Forward ONLY what this build's PackedLayout accepts. ComfyUI
            # 0.34.0 dropped `frame_count` from the signature, and forwarding it
            # blindly raised TypeError on the no-keyframes path -- which is most
            # renders. Introspecting instead of hard-coding means the next
            # signature change degrades rather than crashes.
            base_kw = {"keyframes": None, "refs": refs}
            base_kw = {k: v for k, v in base_kw.items() if k in _BASE_PARAMS}
            if "frame_count" in _BASE_PARAMS:
                base_kw["frame_count"] = frame_count

            # Core no longer passes frame_count either, so a keyframe pinned to
            # the LAST frame would silently never resolve. Derive it: a legal run
            # is 17n+5 and its latent length is 5n+2, so the map is exact.
            if frame_count is None and latent_t >= 2:
                frame_count = (int(latent_t) - 2) // 5 * 17 + 5

            if not keyframes:
                super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                                 **base_kw)
                return

            frame, w_grid = _frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            segments = [("text", text_len)]
            g = torch.zeros(text_len, 3, dtype=torch.float64)
            g[:, 0] = torch.arange(text_len, dtype=torch.float64)
            pos = [g]
            img_pos, img_update = [], []
            audio_pos, audio_update = [], []
            cursor = float(text_len)
            row = text_len
            target_audio_w = (float(w_grid[0]), float(w_grid[-1]))

            # references FIRST, so the cursor lands where the target video starts
            for blk in (refs or []):
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]),
                                               float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

            # keyframes against the post-ref cursor = the target's own timeline
            spans = _video_t_spans(latent_t)
            for kf in keyframes:
                idx = kf["resolved_frame_index"]
                if idx == 0:
                    cond_t = cursor
                elif frame_count is not None and idx == frame_count - 1:
                    cond_t = cursor + sum(spans) - FRAME_RESCALE
                else:
                    # General position, per NikoDemon80/ComfyUI-H3-Motion-Context:
                    # each video token spans FRAME_RESCALE * FRAME_PER_TOKEN[k%5] and
                    # covers FRAME_PER_TOKEN[k%5] pixel frames, so cumulative time at
                    # pixel frame p is exactly FRAME_RESCALE * p. Substituting
                    # p = frame_count-1 reproduces the stock last-frame expression,
                    # which is the proof it is right.
                    #   text_len + FRAME_RESCALE*(fc-1) == text_len + sum(spans) - FRAME_RESCALE
                    # An earlier version here walked the spans cumulatively to the
                    # containing LATENT frame, which is a different (wrong) quantity.
                    cond_t = cursor + FRAME_RESCALE * float(idx)
                kf_t = int(kf.get("latent_t", 1))
                if kf_t > 1:
                    n = kf_t * frame_rows
                    segments.append(("cond", n))
                    pos.append(_video_grid(kf_t, frame, cond_t))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    continue
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

            segments.append(("audio", audio_t * 2))
            pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
            audio_pos.append(torch.arange(row, row + audio_t * 2))
            audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
            row += audio_t * 2

            n_video = latent_t * frame_rows
            segments.append(("video", n_video))
            pos.append(_video_grid(latent_t, frame, cursor))
            img_pos.append(torch.arange(row, row + n_video))
            img_update.append(torch.ones(n_video, dtype=torch.bool))
            row += n_video

            self.seq_len = row
            self.position_ids = torch.cat(pos)
            self.img_pos = torch.cat(img_pos)
            self.img_update = torch.cat(img_update)
            self.audio_pos = torch.cat(audio_pos)
            self.audio_update = torch.cat(audio_update)
            self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
            seg, off = [], 0
            for kind, n in segments:
                seg.append((off, off + n, kind))
                off += n
            self.segments = seg

    M.PackedLayout = LongFormLayout
    _PATCHED = True
    print("[h3_audio] PackedLayout patched: keyframes + references can coexist")
    return True


class H3KeyframeTimeline:
    """Place up to four keyframes anywhere on the clip's timeline.

    Upstream gives you first and last only. This adds an arbitrary position, so you
    can direct "be in THIS pose at 8 seconds" rather than only "start here".

    `cond_video_latents` must be ordered to match the packed sequence — references
    first, then keyframes — because the model concatenates them flat and drops them
    into the non-target image rows in sequence order. This node appends, so run it
    AFTER the reference node.

    A time of -1 disables that slot. Time is in SECONDS and is snapped to the frame
    grid; 0 means the first frame, and anything at or past the clip's end becomes
    the last frame (both are the well-trodden upstream cases).
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "conditioning": ("CONDITIONING",),
            "vae": ("VAE",),
            "length": ("INT", {"default": 141, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "Must match the H3 conditioning node's length. "
                                          "141 = 5.875 s, an AV-aligned run."}),
            "image_1": ("IMAGE",),
            "time_1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 150.0, "step": 0.1,
                                 "tooltip": "Seconds. 0 = first frame. -1 disables."}),
        }
        opt = {}
        for i in (2, 3, 4):
            opt[f"image_{i}"] = ("IMAGE",)
            opt[f"time_{i}"] = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 150.0,
                                          "step": 0.1})
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Add keyframes at arbitrary times, and let them coexist with "
                   "reference images. Run AFTER the H3 reference node. Middle "
                   "positions are experimental; first/last are the proven cases.")

    def go(self, conditioning, vae, length, **kw):
        patch_packed_layout()
        frame_count = snap_run(length)
        entries = []
        for i in (1, 2, 3, 4):
            img = kw.get(f"image_{i}")
            t = kw.get(f"time_{i}", -1.0)
            if img is None or t is None or t < 0:
                continue
            idx = int(round(float(t) * 24.0))
            idx = 0 if idx <= 0 else min(idx, frame_count - 1)
            entries.append((idx, img))
        if not entries:
            return (conditioning,)
        entries.sort(key=lambda e: e[0])

        keyframes = []
        for idx, img in entries:
            z = vae.encode(img[:1])
            keyframes.append({"resolved_frame_index": idx, "latent": z,
                              "latent_t": z.shape[2] if z.dim() == 5 else 1})

        # keep any reference latents the ref node already put in front
        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            refs = meta.get("minimax_refs") or []
            ref_lats = [r["latent"] for r in refs if "latent" in r]
            meta["minimax_keyframes"] = keyframes
            meta["frame_count"] = frame_count
            meta["cond_video_latents"] = ref_lats + [k["latent"] for k in keyframes]
            out.append([cond, meta])
        return (out,)


class H3ReferenceToVideoLongForm(io.ComfyNode):
    """ref2va conditioning with a keyframe that is ALSO SHOWN to the language model.

    This exists because the one thing that cannot be bolted on after the fact is
    PRESENTATION. The stock reference node tokenizes the prompt with its reference
    items inside `clip.tokenize(prompt, minimax_ref_items=...)`; once it returns a
    CONDITIONING, that presentation is fixed. So a keyframe added downstream reaches
    the DiT as a condition row but is never SEEN by Qwen.

    That difference is not cosmetic. In our engine tests the same chain built two
    ways — keyframe as a bare condition row, versus keyframe also presented as the
    next <Picture n> — went from motion that hesitated and reversed direction at
    every join, to joins that read as continuous. A still frame tells the model
    where her arms ARE either way; showing it to the language model is what conveys
    what the pose is DOING.

    Everything else here matches the stock MiniMaxH3ReferenceToVideo: same canvas
    maths, same ref ordering, same encode path. The additions are `keyframe`,
    `keyframe_time`, and `present_keyframe`.

    WHY REFERENCE COUNT IS WORTH SPENDING ON
      A reference image contributes a FIXED number of tokens — one latent frame's
      worth — while the target video grows linearly with duration. So a reference's
      share of the picture tokens collapses as the clip lengthens: 3 references that
      hold 19.4% of a 39-frame render hold 3.2% of a 294-frame one, and at that
      point the source's appearance wins over the reference's. `H3RefBudget` reports
      the number. More references, and `ref_image_size = max` on large sources, are
      the two levers that move it, which is why this node autogrows to nine rather
      than capping at the five it used to offer.

    ORDERING MATTERS. `cond_video_latents` is consumed as a flat concatenation
    dropped into the non-target image rows in sequence order, so references come
    first and keyframes second — matching the patched PackedLayout above. The
    autogrow dict preserves insertion order, and the executor fills it in schema
    order, so iterating `.values()` is the same order stock uses.

    V3 SCHEMA, ON PURPOSE. Autogrow is a `DynamicInput` and has no equivalent in
    the V1 `INPUT_TYPES` dict, so this one node is defined with `io.Schema` while
    the rest of the pack stays V1. Both register through NODE_CLASS_MAPPINGS —
    ComfyUI's loader takes the V1 branch for the module and stores the class
    as-is, and `io.ComfyNode` supplies an `INPUT_TYPES` bridge for everything
    downstream. Do NOT add a `comfy_entrypoint` to this pack: the loader is
    if/elif, so NODE_CLASS_MAPPINGS wins and the entrypoint would never run.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ReferenceToVideoLongForm",
            display_name="H3 Reference to Video (long-form)",
            category=CATEGORY,
            description=("ref2va conditioning where the keyframe is also presented to "
                         "the language model. Chain clips by feeding the previous "
                         "clip's last frame (H3 Chain Frame) into `keyframe`."),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=640, min=32, max=4096, step=32),
                io.Int.Input("height", default=1120, min=32, max=4096, step=32),
                io.Int.Input("length", default=345, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps. 345 = 14.375 s — the longest "
                                     "run inside the trained range that lands exactly "
                                     "on BOTH clocks, 24 fps video and the 40 Hz audio "
                                     "latent. 362 (15.08 s) is a legal video run but "
                                     "rounds the audio, which accumulates across a "
                                     "chain. Aligned runs: 39, 90, 141, 192, 243, "
                                     "294, 345."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="max",
                               tooltip="'match' shrinks anchors to the render's pixel "
                                       "area; 'max' keeps them native up to a 2048 "
                                       "short edge. On a large source 'max' is several "
                                       "times the tokens, which is the point — it is "
                                       "how a reference keeps its share on a long "
                                       "clip. It costs roughly the square of the "
                                       "sequence growth in attention."),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Identity anchor, shown to the language model as "
                                    "the next <Picture n>. Never upscaled; downscaled "
                                    "per ref_image_size."),
                        prefix="ref_image_", min=0, max=MAX_REF_IMAGES)),
                io.Image.Input("keyframe", optional=True,
                               tooltip="Usually the previous clip's last frame."),
                io.Float.Input("keyframe_time", default=0.0, min=0.0, max=150.0,
                               step=0.1, optional=True,
                               tooltip="Seconds. 0 = start here."),
                io.Boolean.Input("present_keyframe", default=True, optional=True,
                                 tooltip="Show the keyframe to the language model as "
                                         "the next <Picture n>. Leave ON — this is "
                                         "what stops motion hesitating at joins."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="max", ref_images=None, keyframe=None,
                keyframe_time=0.0, present_keyframe=True, **kwargs):
        # An Autogrow input is DOCUMENTED to arrive as {name: value}, and stock
        # MiniMaxH3ReferenceToVideo receives it that way. Registered through
        # NODE_CLASS_MAPPINGS rather than comfy_entrypoint it can instead arrive
        # as separate ref_image_N kwargs, which raised
        #   TypeError: execute() got an unexpected keyword argument 'ref_image_1'
        # on ComfyUI 0.34.0. Accept both rather than betting on one: the numeric
        # suffix decides order, so <Picture n> numbering stays stable either way.
        if not ref_images:
            loose = [(k, v) for k, v in kwargs.items()
                     if k.startswith("ref_image_") and v is not None]

            def _order(item):
                tail = item[0].rsplit("_", 1)[-1]
                return int(tail) if tail.isdigit() else 1 << 30

            ref_images = {k: v for k, v in sorted(loose, key=_order)}
        import math
        import comfy.nested_tensor
        import comfy.model_management
        import comfy.utils
        patch_packed_layout()

        frame_count = snap_run(length)
        latent_t = video_latent_t(frame_count)
        audio_t = round(frame_count / 24.0 * 40)
        dev = comfy.model_management.intermediate_device()
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros([1, 24, latent_t, height // 16, width // 16], device=dev),
            torch.zeros([1, 32, 2, audio_t], device=dev)))}

        def resize(image, w, h, crop):
            s = image[..., :3].movedim(-1, 1)
            s = comfy.utils.common_upscale(s, w, h, "lanczos", crop)
            return s.movedim(1, -1)

        ref_items, ref_blocks = [], []
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(32, round(w * scale / 32) * 32)
            th = max(32, round(h * scale / 32) * 32)
            r = resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": r})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": vae.encode(r)})

        keyframes = []
        if keyframe is not None:
            kimg = resize(keyframe[:1], width, height, "disabled")
            idx = int(round(float(keyframe_time) * 24.0))
            idx = 0 if idx <= 0 else min(idx, frame_count - 1)
            keyframes.append({"resolved_frame_index": idx, "latent": vae.encode(kimg),
                              "latent_t": 1})
            if present_keyframe:
                # presentation ONLY — no ref block, so it stays a keyframe to the DiT
                # while the LM still sees it. This is the whole point of the node.
                ref_items.append({"type": "image", "data": kimg})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if keyframes:
            values["minimax_keyframes"] = keyframes
            values["frame_count"] = frame_count
        if ref_blocks or keyframes:
            values["cond_video_latents"] = ([r["latent"] for r in ref_blocks]
                                            + [k["latent"] for k in keyframes])
            cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


NODE_CLASS_MAPPINGS = {
    "H3KeyframeTimeline": H3KeyframeTimeline,
    "H3ReferenceToVideoLongForm": H3ReferenceToVideoLongForm,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KeyframeTimeline": "H3 Keyframe Timeline",
    "H3ReferenceToVideoLongForm": "H3 Reference to Video (long-form)",
}
