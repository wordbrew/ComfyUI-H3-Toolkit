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
import logging

import torch

import comfy.model_management
import node_helpers
from comfy_api.latest import io

from .avlatent import av
from .chunkplan import snap_context
from .timing import FRAME_PER_TOKEN, snap_run, video_latent_t

CATEGORY = "MiniMax H3/video"

# Mirrors stock MiniMaxH3ReferenceToVideo's autogrow ceiling. Nine matters: a
# reference's share of the picture tokens falls as the clip lengthens, and nine
# large references on `max` is what holds a 294-frame render above the ratio a
# 39-frame one gets from three. See H3RefBudget.
MAX_REF_IMAGES = 9

# Stock's ceilings for the other three kinds. A reference VIDEO costs a whole
# clip's worth of tokens per block, which is why it caps far lower than images.
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

# Core's own constants and canvas maths, imported so a change upstream follows
# rather than drifting. A reference video is sized on the SAME canvas rule the
# render uses, not on the render's own width/height, because its aspect is its
# own; getting that wrong puts the reference on a stretched grid and the model
# reads the distortion as part of the look.
try:
    from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE,
                                               REF_IMAGE_SHORT_EDGE,
                                               adapt_canvas)
except Exception:  # pragma: no cover - core moved or renamed
    CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, adapt_canvas = 32, 2048, None

# 24 fps video, 40 Hz audio latent, and Qwen is shown a reference video at 2 fps.
FPS = 24
QWEN_REF_FPS = 2


def encode_ref_audio(audio_vae, audio):
    """Waveform -> ([1, 32, 2, T], T). Resamples to the audio VAE's own rate.

    Core has this as a private helper. Prefer core's if it is there, so a change
    upstream follows; the local copy is the same six lines and keeps the pack
    working against a build that renames it.
    """
    fn = None
    try:
        from comfy_extras.nodes_minimax_h3 import _encode_ref_audio as fn
    except Exception:  # pragma: no cover - core moved or renamed
        fn = None
    if fn is not None:
        return fn(audio_vae, audio)
    waveform = audio["waveform"]                    # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


#  layout patch: let keyframes and references coexist
# --------------------------------------------------------------------------- #

_PATCHED = False

# WINDOW POSITION HAND-OFF
#
# `PackedLayout` is built inside `extra_conds`, which receives the payload but
# passes no window position to the constructor. Rather than patch that call site
# in core -- which a ComfyUI update silently reverts -- `windowing.py` leaves the
# value here and the next layout build takes it.
#
# CONSUME-ONCE. The hook sets this per window immediately before the model call;
# __init__ reads it and resets to 0, so a build nobody set it for behaves as an
# unwindowed render and a stale value cannot survive into a later one.
#
# THE THING THAT MUST NOT CHANGE: the signature stays a 5-tuple. `_forward`
# reuses the prebuilt layout only when the signature matches what IT computes,
# and appending window_start to it makes every offset window miss, rebuild
# without the offset (already consumed), and silently revert to origin
# positioning -- correct on window 0, wrong on all the rest. Two windows cannot
# share a layout anyway: each carries its own payload copy.
_WINDOW_START = [0]


def set_window_start(frames):
    """Position of the window about to be built, in PIXEL frames."""
    _WINDOW_START[0] = int(frames or 0)


def take_window_start():
    v = _WINDOW_START[0]
    _WINDOW_START[0] = 0
    return v


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

    def _ref_t_span(blk):
        kind = blk["kind"]
        if kind == "image":
            return 1.0
        if kind == "audio":
            return float(blk["ref_audio_t"])
        if kind in ("video", "video_audio"):
            return max(float(blk["ref_audio_t"]),
                       sum(_video_t_spans(blk["latent_t"])))
        return 0.0

    _ref_t_span = getattr(M, "_ref_t_span", _ref_t_span)

    class LongFormLayout(Base):
        _h3_longform_patched = True

        def _offset_target(self, window_start):
            """Move the TARGET rows to where this window sits on the clip.

            `cursor` is text_len plus the reference spans and does not depend on
            the window, so every window is otherwise told it begins at the clip
            origin and renders the opening of the shot; the overlaps then
            crossfade between two openings, which is the flicker. Measured
            2026-08-28: 167 layout builds, one cursor.

            Shifting column 0 of the finished position table is exactly what a
            later cursor produces -- `_video_grid` writes the time coordinate
            there and nothing else reads it -- and needs no change to core,
            which is the point.

            References and keyframes are deliberately NOT moved: they sit
            relative to the clip origin either way. Only the target belongs at
            the window's own place on the timeline.
            """
            if not window_start:
                return
            off = FRAME_RESCALE * float(window_start)
            for a, b, kind in self.segments:
                if kind in ("audio", "video"):
                    self.position_ids[a:b, 0] += off

        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None,
                     window_start=0):
            # Forward ONLY what this build's PackedLayout accepts. ComfyUI
            # 0.34.0 dropped `frame_count` from the signature, and forwarding it
            # blindly raised TypeError on the no-keyframes path -- which is most
            # renders. Introspecting instead of hard-coding means the next
            # signature change degrades rather than crashes.
            base_kw = {"keyframes": None, "refs": refs}
            base_kw = {k: v for k, v in base_kw.items() if k in _BASE_PARAMS}
            if "frame_count" in _BASE_PARAMS:
                base_kw["frame_count"] = frame_count
            # NOT forwarded to the base: the offset is applied to the finished
            # position table instead (see _offset_target), which is what lets
            # this work on stock ComfyUI with no core edit at all.
            if window_start == 0:
                window_start = take_window_start()
            # debug, not info: one per layout build is ~120 lines on a 3-window
            # run. Kept because "did the window position reach the build that is
            # actually used" is the question this whole conversion turns on.
            logging.debug("H3 layout build: latent_t %s | window_start %s | "
                          "keyframes %s | refs %s", latent_t, window_start,
                          len(keyframes or ()), len(refs or ()))

            # Core no longer passes frame_count either, so a keyframe pinned to
            # the LAST frame would silently never resolve. Derive it: a legal run
            # is 17n+5 and its latent length is 5n+2, so the map is exact.
            if frame_count is None and latent_t >= 2:
                frame_count = (int(latent_t) - 2) // 5 * 17 + 5

            if not keyframes:
                super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                                 **base_kw)
                self._offset_target(window_start)
                return

            frame, w_grid = _frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            segments = [("text", text_len)]
            g = torch.zeros(text_len, 3, dtype=torch.float64)
            g[:, 0] = torch.arange(text_len, dtype=torch.float64)
            pos = [g]
            img_pos, img_update = [], []
            audio_pos, audio_update = [], []
            row = text_len
            target_audio_w = (float(w_grid[0]), float(w_grid[-1]))

            # The target timeline starts after the reference SPANS, but the
            # reference ROWS are emitted after the keyframe rows. Those are two
            # different orders and conflating them is a silent corruption:
            # `_cond_video_rows` concatenates the cond latents flat and the model
            # scatters them with `all_video_rows[~img_update] = cond_video_rows`,
            # so row order has to match the order core BUILDS that list in --
            # keyframes first, then refs (model_base.py:2201,2206). This class
            # emitted refs first until 2026-08-30, which fed every ref latent into
            # a keyframe's row slot whenever a graph used both. Stock PackedLayout
            # has always done it this way; the divergence was ours.
            cursor = float(text_len)
            for blk in (refs or []):
                cursor += _ref_t_span(blk)

            # keyframes against the post-ref cursor = the target's own timeline
            spans = _video_t_spans(latent_t)
            for kf in keyframes:
                idx = kf["resolved_frame_index"]
                if idx == 0:
                    cond_t = cursor
                elif frame_count is not None and idx == frame_count - 1:
                    cond_t = cursor + sum(spans) - FRAME_RESCALE
                else:
                    # General position: each video token spans
                    # FRAME_RESCALE * FRAME_PER_TOKEN[k%5] and covers
                    # FRAME_PER_TOKEN[k%5] pixel frames, so cumulative time at
                    # pixel frame p is exactly FRAME_RESCALE * p. Substituting
                    # p = frame_count-1 reproduces the stock last-frame expression,
                    # which is the proof it is right.
                    #   text_len + FRAME_RESCALE*(fc-1) == text_len + sum(spans) - FRAME_RESCALE
                    # An earlier version here walked the spans cumulatively to the
                    # containing LATENT frame, which is a different (wrong) quantity.
                    cond_t = cursor + FRAME_RESCALE * float(idx)
                kf_t = int(kf.get("latent_t", 1))
                if kf.get("latent") is not None:
                    if kf_t > 1:
                        n = kf_t * frame_rows
                        segments.append(("cond", n))
                        pos.append(_video_grid(kf_t, frame, cond_t))
                    else:
                        n = frame_rows
                        g = torch.empty(frame_rows, 3, dtype=torch.float64)
                        g[:, 0] = cond_t
                        g[:, 1:] = frame
                        segments.append(("cond", frame_rows))
                        pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                # a keyframe may carry audio too; stock emits those rows and core
                # puts their latents in cond_audio_latents, so the same rule holds
                kf_audio = kf.get("audio_latent")
                if kf_audio is not None:
                    rt = kf_audio.shape[-1]
                    segments.append(("cond_audio", rt * 2))
                    pos.append(_audio_grid(cond_t, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

            # then the reference rows, walking their own cursor from the text
            rcursor = float(text_len)
            for blk in (refs or []):
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = rcursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    rcursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(rcursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    rcursor += float(rt)
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(rcursor, rt, float(r_w_grid[0]),
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
            # LAST: it walks self.segments, so it cannot run before the segment
            # table exists. It did once, on this branch only, and every windowed
            # render with a keyframe raised AttributeError.
            self._offset_target(window_start)

    M.PackedLayout = LongFormLayout
    _PATCHED = True
    print("[h3_audio] PackedLayout patched: keyframes + references can coexist")
    return True


class H3KeyframeTimeline:
    """Place up to four keyframes anywhere on the clip's timeline.

    Upstream gives you first and last only. This adds an arbitrary position, so you
    can direct "be in THIS pose at 8 seconds" rather than only "start here".

    `cond_video_latents` must be ordered to match the packed sequence — KEYFRAMES
    first, then references — because the model concatenates them flat and drops
    them into the non-target image rows in sequence order. That is core's order
    (model_base.py:2201,2206) and core rebuilds the list itself, so the copy this
    node writes is belt-and-braces; what actually decides anything is the ROW
    order in the patched layout. This node appends to whatever the reference node
    already set, so run it AFTER that node.

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
            meta["cond_video_latents"] = [k["latent"] for k in keyframes] + ref_lats
            out.append([cond, meta])
        return (out,)


class H3MotionContext:
    """Put the previous chunk's tail on THIS chunk's timeline, as conditioning.

    THE MECHANISM, AND WHY IT IS NOT THE PIN
      The rows the model sees are identical between a reference and a keyframe;
      only their TIME COORDINATES differ, and the coordinates are what say
      "separate clip" versus "this clip, earlier". A reference sits in its own
      cursor region and reads as another clip. The same rows placed on the
      target's own timeline read as this clip's past.

      So this is NOT H3 Latent Pin and does not replace it. The pin writes the
      previous tail INTO the target and masks it to 0 -- those frames become part
      of the target, are reproduced rather than generated, and whatever the model
      adds to the rest is inherited by the next chunk. This adds cond ROWS. The
      target still generates all of its own frames; the tail is guidance.

      Measured engine-side (085, three arms, three links):

          A  keyframe only        background drift by link 3   -0.0183
          B  + motion context 22                               +0.0007

      Context frames HOLD brightness where a keyframe alone drifts.

    ENCODE MODE, WHICH IS THE PART THAT WAS WRONG FOR NINE WAVES
      The tail is encoded in ONE VAE call with the batch axis as time, and each
      resulting latent STEP is placed at its own pixel offset. Encoding frame by
      frame gives stills with no temporal structure, and slicing a previous run's
      latent is worse still: FRAME_PER_TOKEN is (1,4,4,4,4), so coverage is
      POSITIONAL, and steps taken from the end of a 52-step latent carry the
      structure of positions 45..51 while landing at 0..K-1. Measured in 087 as
      HEAD 12.5 against 3.8 and joins 7-21 against 3-5.

    ONLY 39, 22, 5 AND 1 ARE DISTINCT
      VIDEO_RUN_GRID. Every context test before 2026-08-09 used 12, which snaps
      DOWN to 5 and then covers the FIRST five of the twelve, so the pinned run
      ended seven frames early. That systematic offset was in 085 and 087 and
      explains their widened joins better than anything claimed at the time.

    RUN IT AFTER THE REFERENCE NODE. Anchors and motion context coexist -- an
    earlier claim that they did not was wrong -- but `cond_video_latents` has to
    stay in packed-sequence order, so this appends rather than replaces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING", {"tooltip": "From the H3 reference "
                             "node. Run this AFTER it."}),
            "vae": ("VAE", {"tooltip": "The video VAE."}),
            "length": ("INT", {"default": 141, "min": 5, "max": 3600, "step": 17,
                       "tooltip": "This chunk's run. Wire H3 Chunk Open's "
                                  "`length`."}),
        }, "optional": {
            "context_images": ("IMAGE", {"tooltip": "The PREVIOUS chunk's tail, "
                               "in pixels — H3 Chunk Open's `context`. Unwired, "
                               "or on the first chunk, this passes the "
                               "conditioning through untouched."}),
            "context_frames": (["22", "39", "5", "1", "0"], {"default": "22",
                               "tooltip": "Frames of tail to place. Only these "
                                          "encode to distinct VAE runs. 22 is "
                                          "what the validated long-form recipe "
                                          "used; 39 is the smallest run on both "
                                          "the video and audio clocks."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "context_frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    # SUPERSEDED 2026-08-30. Downstream of the reference node it cannot present
    # the tail to the language model, and without that every chunk after the
    # first rendered a reference image instead of the shot. Use
    # H3ReferenceToVideoLongForm's own `context_images` / `context_frames`. Kept
    # registered so saved graphs still load; DEPRECATED hides it from the menu.
    DEPRECATED = True
    DESCRIPTION = ("Place the previous chunk's tail as cond rows on this chunk's "
                   "own timeline, so the model reads it as this clip earlier "
                   "rather than as another clip.")

    def go(self, conditioning, vae, length, context_images=None,
           context_frames="22"):
        from .chunkplan import snap_context
        from .timing import FRAME_PER_TOKEN, snap_run, video_latent_t

        n = snap_context(int(context_frames))
        if context_images is None or n <= 0:
            why = ("no context wired — first chunk, or nothing to carry"
                   if context_images is None else "context_frames 0")
            return (conditioning, 0, f"H3 MOTION CONTEXT: passthrough ({why})")

        patch_packed_layout()
        frame_count = snap_run(length)
        have = int(context_images.shape[0])
        if have < n:
            n = snap_context(have)
            if n <= 0:
                return (conditioning, 0,
                        f"H3 MOTION CONTEXT: only {have} frame(s) available, "
                        f"fewer than the smallest distinct run")
        if n >= frame_count:
            raise ValueError(
                f"H3 Motion Context: {n} context frames do not fit inside a "
                f"{frame_count}-frame chunk. Shorten the context or lengthen "
                f"the chunk.")

        # ONE call, batch axis as time. The tail arrives at the render's own size
        # because it is the previous chunk's decode, so there is nothing to
        # resize -- and resizing here would be a second resample of content that
        # has already been through the VAE once.
        tail = context_images[-n:]
        z = vae.encode(tail)
        steps = int(z.shape[2]) if z.dim() == 5 else 1

        # step k covers FRAME_PER_TOKEN[k % 5] pixel frames, so its own offset is
        # the sum of everything before it -- 0, 1, 5, 9, 13, 17, 18, ...
        offsets, at = [], 0
        for k in range(steps):
            offsets.append(at)
            at += FRAME_PER_TOKEN[k % 5]

        rows = [{"resolved_frame_index": min(o, frame_count - 1),
                 "latent": z[:, :, k:k + 1], "latent_t": 1}
                for k, o in enumerate(offsets)]

        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            refs = meta.get("minimax_refs") or []
            ref_lats = [r["latent"] for r in refs if "latent" in r]
            kfs = list(meta.get("minimax_keyframes") or [])
            kfs.extend(rows)                      # coexist, never replace
            meta["minimax_keyframes"] = kfs
            meta["frame_count"] = frame_count
            meta["cond_video_latents"] = [k["latent"] for k in kfs] + ref_lats
            out.append([cond, meta])

        info = (f"H3 MOTION CONTEXT: {n} frame(s) placed as {steps} cond row(s) "
                f"at pixel offsets {offsets[:6]}{'...' if steps > 6 else ''}\n"
                f"  on THIS chunk's timeline, so they read as its past rather "
                f"than as another clip\n"
                f"  {len(rows)} row(s) added to {len(kfs) - len(rows)} existing "
                f"keyframe(s), alongside {len(ref_lats)} reference(s)")
        logging.info("H3MotionContext: %d frames -> %d rows", n, steps)
        return {"ui": {"h3char": [info]}, "result": (out, n, info)}


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

    ORDERING MATTERS. The cond latents are consumed as a flat concatenation
    dropped into the non-target image rows in sequence order, and core builds that
    list keyframes first, references second — which is the order the patched
    PackedLayout above emits rows in. Within the references, the autogrow dict
    preserves insertion order and the executor fills it in schema order, so
    iterating `.values()` is the same order stock uses.

    REFERENCE KINDS. All four of stock's are here: images, videos, a video's own
    soundtrack, and standalone audio. The audio one is not an afterthought — it
    is the only thing that pins VOICE, and without it a chained dialogue take
    invents the timbre afresh in every chunk.

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
                io.Image.Input("context_images", optional=True,
                               tooltip="MOTION CONTEXT: the previous chunk's tail, "
                                       "in pixels. H3 Chunk Open's `context`. Its "
                                       "last N frames are placed as cond rows on "
                                       "THIS clip's own timeline, so the model "
                                       "reads them as this clip earlier rather "
                                       "than as another clip. It has to happen "
                                       "HERE and not downstream: the last frame is "
                                       "also presented to the language model, and "
                                       "presentation is fixed once this node has "
                                       "tokenized."),
                io.Combo.Input("context_frames", options=["0", "1", "5", "22", "39"],
                               default="0", optional=True,
                               tooltip="Frames of tail to place. 0 is off and is "
                                       "the default, so nothing changes unless you "
                                       "ask for it. Only 1/5/22/39 encode to "
                                       "distinct VAE runs — an off-grid count "
                                       "snaps DOWN and then covers the FIRST "
                                       "frames of what it was given, so the pinned "
                                       "run ends early and the join jumps. 39 is "
                                       "the smallest run on both the video and "
                                       "audio clocks."),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Reference video at 24 fps, 2-15 s. Shown to "
                                    "the language model at 2 fps with timestamps, "
                                    "and to the DiT as its own clip. A whole clip "
                                    "of tokens, so it is expensive — use it for "
                                    "an action or a look you cannot describe."),
                        prefix="ref_video_", min=0, max=MAX_REF_VIDEOS)),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the SAME-NUMBERED reference "
                                    "video. ref_video_audio_2 belongs to "
                                    "ref_video_2; the pairing is by number, not "
                                    "by position."),
                        prefix="ref_video_audio_", min=0, max=MAX_REF_VIDEOS)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            tooltip="VOICE REFERENCE. A few seconds of clean "
                                    "speech pins the timbre, which is otherwise "
                                    "invented afresh in every chunk of a chained "
                                    "take. It is presented as the next <Audio n>, "
                                    "so name it in the prompt to bind it to a "
                                    "speaker."),
                        prefix="ref_audio_", min=0, max=MAX_REF_AUDIOS)),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="max", ref_images=None, keyframe=None,
                keyframe_time=0.0, present_keyframe=True, context_images=None,
                context_frames="0", ref_videos=None, ref_video_audios=None,
                ref_audios=None, **kwargs):
        # An Autogrow input is DOCUMENTED to arrive as {name: value}, and stock
        # MiniMaxH3ReferenceToVideo receives it that way. Registered through
        # NODE_CLASS_MAPPINGS rather than comfy_entrypoint it can instead arrive
        # as separate ref_image_N kwargs, which raised
        #   TypeError: execute() got an unexpected keyword argument 'ref_image_1'
        # on ComfyUI 0.34.0. Accept both rather than betting on one: the numeric
        # suffix decides order, so <Picture n> numbering stays stable either way.
        def _order(item):
            tail = item[0].rsplit("_", 1)[-1]
            return int(tail) if tail.isdigit() else 1 << 30

        def _collect(given, prefix):
            if given:
                return given
            # `ref_video_` also prefixes `ref_video_audio_`, so match the SUFFIX
            # shape too: a bare number after the prefix, nothing else.
            loose = [(k, v) for k, v in kwargs.items()
                     if k.startswith(prefix) and v is not None
                     and k[len(prefix):].isdigit()]
            return {k: v for k, v in sorted(loose, key=_order)}

        ref_images = _collect(ref_images, "ref_image_")
        ref_videos = _collect(ref_videos, "ref_video_")
        ref_video_audios = _collect(ref_video_audios, "ref_video_audio_")
        ref_audios = _collect(ref_audios, "ref_audio_")
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

        # REFERENCE VIDEOS. Sized on the canvas rule rather than the render's
        # own width and height — the reference has its own aspect, and forcing
        # it onto the target's would teach the model the distortion. Matches
        # stock MiniMaxH3ReferenceToVideo block for block.
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get(
                "ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            if adapt_canvas is None:  # pragma: no cover - core moved or renamed
                cw, ch = width, height
            else:
                cw, ch = adapt_canvas(vw, vh)
                if vw * vh < cw * ch:
                    # never upscale a small reference into a big canvas
                    cw = max(CANVAS_MULTIPLE,
                             round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                    ch = max(CANVAS_MULTIPLE,
                             round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = int(frames.shape[0])
            if n < 5:
                raise ValueError(
                    f"H3 Reference to Video: {name} has {n} frame(s). A "
                    f"reference video needs at least 5 (~0.2 s at 24 fps).")
            while n % 17 != 5:                      # a legal run, trimmed DOWN
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = None, 0
            if soundtrack is not None:
                audio_latent, ref_audio_t = encode_ref_audio(audio_vae, soundtrack)
                # the soundtrack gets its own <Audio j>, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps, not every frame
            sample_idx = list(range(0, n, FPS // QWEN_REF_FPS))
            ref_items.append({
                "type": "video", "data": frames[sample_idx],
                "timestamps": [i / float(QWEN_REF_FPS)
                               for i in range(len(sample_idx))]})
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2], "latent_h": ch // 16,
                "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                "latent": z, "audio_latent": audio_latent})

        # STANDALONE REFERENCE AUDIO — the voice pin. No video rows, so it is
        # cheap next to a reference video, and it is the only thing that stops a
        # chained dialogue take reinventing the timbre at every chunk.
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

        keyframes = []
        n_ctx = 0
        if context_images is not None and int(context_frames) > 0:
            # MOTION CONTEXT. The rows the model sees are identical between a
            # reference and a keyframe; only their TIME COORDINATES differ, and
            # the coordinates are what say "separate clip" versus "this clip,
            # earlier". Reference rows sit in their own cursor region and read as
            # another clip -- the ping-pong nine waves chased. The same rows on
            # the target's own timeline read as its past.
            #
            # IT HAS TO HAPPEN HERE. Built as a node downstream of this one it
            # cannot present the tail to the language model, because `tokenize`
            # below has already fixed the presentation. Tried on 2026-08-30: the
            # DiT got seven context rows, Qwen saw only the references, and every
            # chunk after the first rendered a reference image instead of the
            # shot.
            avail = int(context_images.shape[0])
            n_ctx = snap_context(min(int(context_frames), avail))
            if n_ctx >= frame_count:
                raise ValueError(
                    f"H3 Reference to Video: {n_ctx} context frames do not fit "
                    f"inside a {frame_count}-frame clip. The pinned run has to be "
                    f"a small fraction of the timeline.")
            if n_ctx >= 1:
                tail = resize(context_images[avail - n_ctx:], width, height,
                              "disabled")
                # ONE call, batch axis as time. Per-frame calls give stills with
                # no temporal structure, and slicing a PREVIOUS RUN's latent is
                # worse again -- FRAME_PER_TOKEN is (1,4,4,4,4) so coverage is
                # positional, and steps from the end of a 52-step latent carry
                # the structure of positions 45..51 while landing at 0..K-1.
                # Measured in 087: HEAD 12.5 against 3.8, joins 7-21 against 3-5.
                enc = vae.encode(tail)
                n_steps = int(enc.shape[2]) if enc.dim() == 5 else 1
                off, acc = [], 0
                for k in range(n_steps):
                    off.append(acc)
                    acc += FRAME_PER_TOKEN[k % 5]
                if acc != n_ctx:
                    raise RuntimeError(
                        f"H3 Reference to Video: {n_ctx} frames encoded to "
                        f"{n_steps} step(s) covering {acc}. The VAE grid is not "
                        f"what this assumes; refusing to place rows that would "
                        f"land at the wrong times.")
                for k in range(n_steps):
                    keyframes.append({"resolved_frame_index": off[k],
                                      "latent": enc[:, :, k:k + 1],
                                      "latent_t": 1})
                if present_keyframe:
                    ref_items.append({"type": "image", "data": tail[-1:]})

        if keyframe is not None and not keyframes:
            # Skipped when context is present: a context block already anchors
            # frame 0 onward, so a keyframe there would double-pin it. Engine's
            # rule, and the reason context and keyframe are alternatives rather
            # than additions.
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
            # Core's order — keyframes, then references — and a kind "audio"
            # block carries no video latent at all, so both lists are built by
            # what is present rather than by position.
            values["cond_video_latents"] = (
                [k["latent"] for k in keyframes if k.get("latent") is not None]
                + [r["latent"] for r in ref_blocks if r.get("latent") is not None])
            aud = ([k["audio_latent"] for k in keyframes
                    if k.get("audio_latent") is not None]
                   + [r["audio_latent"] for r in ref_blocks
                      if r.get("audio_latent") is not None])
            if aud:
                values["cond_audio_latents"] = aud
            cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)



NODE_CLASS_MAPPINGS = {
    "H3KeyframeTimeline": H3KeyframeTimeline,
    "H3MotionContext": H3MotionContext,
    "H3ReferenceToVideoLongForm": H3ReferenceToVideoLongForm,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KeyframeTimeline": "H3 Keyframe Timeline",
    "H3MotionContext": "H3 Motion Context (this clip, earlier)",
    "H3ReferenceToVideoLongForm": "H3 Reference to Video (long-form)",
}
