"""MiniMax-H3 long-form video nodes — keyframe timeline, audio lock, latent pin.

Companion to the audio prompt nodes in this pack. Everything here came out of the
engine work in ~/projects/h3 (see docs/long-form-waves.md); these are the pieces
that do NOT exist in ComfyUI's stock H3 nodes.

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
extrapolation of the same positional maths, not a measured result. Treat a middle
mark as experimental until you have looked at one.

WHAT WE LEARNED THE HARD WAY (docs/long-form-waves.md has the full log)
  - The LATENT PIN cuts. Every join cut we chased for nine rounds sat exactly at the
    frame where pinned content stopped. Softening, feathering, gradients and
    sigma-release do not move it. H3LatentPin is here because it is a legitimate
    inpainting tool, NOT because it works for continuation.
  - A REFERENCE VIDEO prevents the cut but acts as a destination, so the clip
    converges onto it (ping-pong). The two effects were never separable.
  - A KEYFRAME is bound to frame 0 positionally, so it cannot act as a destination.
    That is why it is the continuation mechanism that works.
  - AUDIO LOCK has no such problem: those cuts all sat at a temporal mask EDGE, and
    locking a full clip end to end has no edge.
"""

import logging
import torch

import comfy.model_management
import node_helpers

CATEGORY = "MiniMax H3/video"


def _av(samples):
    """(video, audio) out of an H3 AV latent.

    NestedTensor.__getitem__ BROADCASTS the index into every contained tensor
    rather than selecting one, so `samples[0]` / `samples[1]` silently does the
    wrong thing and then IndexErrors. The tensors are reached via `.tensors`.
    """
    t = getattr(samples, "tensors", None)
    if t is None:
        t = samples.unbind() if hasattr(samples, "unbind") else samples
    return t[0], t[1]



# --------------------------------------------------------------------------- #
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
            if not keyframes:
                super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                                 keyframes=None, refs=refs, frame_count=frame_count)
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


# --------------------------------------------------------------------------- #
#  nodes
# --------------------------------------------------------------------------- #

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
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "Must match the H3 conditioning node's length."}),
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
        frame_count = _align(length)
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


class H3AudioLock:
    """Generate video against a soundtrack the model cannot change.

    Replaces the AV latent's audio half with a real track and masks the sampler off
    there, so only video is generated. Both streams share one packed sequence, so
    clamping audio every step makes the picture answer to it.

    This is how you get continuous music across a chain of clips: generate the whole
    soundtrack in ONE pass at a tiny canvas (audio latent length depends on DURATION,
    not resolution, so 32x32 makes a 90-second render cheap), then hand each clip its
    own SLICE. Pinning the same block into every clip makes the music restart at
    every join — that was measured, and it is not subtle.

    Unlike the video pin this introduces no seam: every join cut we ever traced sat
    at a temporal mask edge, and locking the full clip has no edge.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "audio_vae": ("VAE",),
            "audio": ("AUDIO",),
            "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0,
                                         "step": 0.01,
                                         "tooltip": "Where in the track this clip starts. "
                                                    "Chain clips by advancing this."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "1.0 pins the audio exactly. Lower lets "
                                              "the model reinterpret it."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Pin a real soundtrack into the AV latent so only video is "
                   "generated. Advance offset_seconds per clip to keep music "
                   "continuous across a chain.")

    def go(self, latent, audio_vae, audio, offset_seconds, strength):
        import comfy.nested_tensor
        video, aud = _av(latent["samples"])
        target_t = aud.shape[-1]

        wav = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, vae_sr)
        z = audio_vae.encode(wav[:1].movedim(1, -1))          # [1, 32, 2, T]

        start = int(round(float(offset_seconds) * 40))        # 40 Hz audio latent
        z = z[..., start:start + target_t]
        if z.shape[-1] < target_t:                            # ran off the end
            # NOT F.pad(mode="replicate"): on a 4D tensor that needs pads for the
            # last TWO dims and raises. Tile the final audio latent frame instead.
            pad = target_t - z.shape[-1]
            z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], pad)], dim=-1)
        z = z.to(aud.device, aud.dtype)

        mask_v = torch.ones_like(video)
        mask_a = torch.full_like(aud, 1.0 - float(strength))
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        return (out,)


class H3LatentPin:
    """Seed a clip's opening with a previous clip's latents (temporal outpaint).

    HONEST WARNING, from nine rounds of measurement: this DOES NOT give a seamless
    continuation. The model reproduces the pinned frames and then splices to its own
    scene at exactly the frame where the pin ends — a visible cut. Feathering,
    partial strength, whole-clip gradients and sigma-release all fail to move it.
    For joining clips use H3KeyframeTimeline instead, which is positionally bound to
    frame 0 and has no mask edge.

    It is included because temporal/region pinning is a legitimate tool for other
    jobs — holding an opening steady, inpainting a span, style continuity where a
    cut does not matter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "previous_latent": ("LATENT",),
            "overlap_frames": ("INT", {"default": 5, "min": 5, "max": 120, "step": 1,
                                       "tooltip": "Pixel frames of the previous clip's "
                                                  "tail to pin. Quantised to the latent "
                                                  "grid: 5->2, 22->7, 39->12."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Pin the previous clip's tail into this clip's opening. Expect a "
                   "visible cut where the pin ends — see the node description.")

    def go(self, latent, previous_latent, overlap_frames, strength):
        import comfy.nested_tensor
        video, aud = _av(latent["samples"])
        pv, pa = _av(previous_latent["samples"])

        kv = min(_video_latent_t(max(5, overlap_frames)), video.shape[2] - 1)
        ka = min(int(round(overlap_frames / 24.0 * 40)), aud.shape[-1] - 1)

        new_v = video.clone()
        new_a = aud.clone()
        new_v[:, :, :kv] = pv[:, :, -kv:].to(new_v.device, new_v.dtype)
        new_a[..., :ka] = pa[..., -ka:].to(new_a.device, new_a.dtype)

        mask_v = torch.ones_like(video)
        mask_a = torch.ones_like(aud)
        mask_v[:, :, :kv] = 1.0 - float(strength)
        mask_a[..., :ka] = 1.0 - float(strength)

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((new_v, new_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        return (out,)


class H3ChainFrame:
    """Take the last frame of a rendered clip, ready to keyframe the next one.

    The whole of the long-form recipe in one node: hand this to
    H3KeyframeTimeline at time 0 on the next clip. Deliberately PIXELS, not
    latents — handing the previous run's latent straight over sounds lossless but
    is out of distribution (a video's final latent frame carries several pixel
    frames of motion plus causal conv state, where the keyframe slot expects a
    single-image encode) and the error compounds every link.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "from_end": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1,
                                 "tooltip": "1 = the very last frame."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = "Last frame of a clip, for keyframing the next one."

    def go(self, images, from_end):
        i = max(0, images.shape[0] - int(from_end))
        return (images[i:i + 1],)


# --------------------------------------------------------------------------- #

def _align(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


NODE_CLASS_MAPPINGS = {
    "H3KeyframeTimeline": H3KeyframeTimeline,
    "H3AudioLock": H3AudioLock,
    "H3LatentPin": H3LatentPin,
    "H3ChainFrame": H3ChainFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KeyframeTimeline": "H3 Keyframe Timeline",
    "H3AudioLock": "H3 Audio Lock",
    "H3LatentPin": "H3 Latent Pin (cuts — read description)",
    "H3ChainFrame": "H3 Chain Frame",
}


# --------------------------------------------------------------------------- #
#  the reference node we actually need
# --------------------------------------------------------------------------- #

class H3ReferenceToVideoLongForm:
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

    ORDERING MATTERS. `cond_video_latents` is consumed as a flat concatenation
    dropped into the non-target image rows in sequence order, so references come
    first and keyframes second — matching the patched PackedLayout above.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "width": ("INT", {"default": 640, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 1120, "min": 32, "max": 4096, "step": 32}),
            "length": ("INT", {"default": 362, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "Frames at 24 fps. 362 = 15.08 s, the top of "
                                          "the trained range."}),
            "ref_image_size": (["match", "max"], {"default": "max",
                               "tooltip": "'match' shrinks anchors to the render's pixel "
                                          "area; 'max' keeps them near full size for "
                                          "identity. 'max' costs ~1.8x."}),
        }
        opt = {f"ref_image_{i}": ("IMAGE",) for i in range(1, 6)}
        opt.update({
            "keyframe": ("IMAGE", {"tooltip": "Usually the previous clip's last frame."}),
            "keyframe_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 150.0,
                                        "step": 0.1,
                                        "tooltip": "Seconds. 0 = start here."}),
            "present_keyframe": ("BOOLEAN", {"default": True,
                                 "tooltip": "Show the keyframe to the language model as "
                                            "the next <Picture n>. Leave ON — this is "
                                            "what stops motion hesitating at joins."}),
        })
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("ref2va conditioning where the keyframe is also presented to the "
                   "language model. Chain clips by feeding the previous clip's last "
                   "frame (H3 Chain Frame) into `keyframe`.")

    def go(self, clip, vae, audio_vae, prompt, width, height, length, ref_image_size,
           **kw):
        import math
        import comfy.nested_tensor
        import comfy.model_management
        import comfy.utils
        patch_packed_layout()

        frame_count = _align(length)
        latent_t = _video_latent_t(frame_count)
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
        for i in range(1, 6):
            img = kw.get(f"ref_image_{i}")
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, 2048 / min(w, h))
            tw = max(32, round(w * scale / 32) * 32)
            th = max(32, round(h * scale / 32) * 32)
            r = resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": r})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": vae.encode(r)})

        keyframes = []
        kf = kw.get("keyframe")
        if kf is not None:
            kimg = resize(kf[:1], width, height, "disabled")
            idx = int(round(float(kw.get("keyframe_time", 0.0)) * 24.0))
            idx = 0 if idx <= 0 else min(idx, frame_count - 1)
            keyframes.append({"resolved_frame_index": idx, "latent": vae.encode(kimg),
                              "latent_t": 1})
            if kw.get("present_keyframe", True):
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
        return (cond, latent)


NODE_CLASS_MAPPINGS["H3ReferenceToVideoLongForm"] = H3ReferenceToVideoLongForm
NODE_DISPLAY_NAME_MAPPINGS["H3ReferenceToVideoLongForm"] = "H3 Reference to Video (long-form)"


class H3MaskInpaint:
    """Replace a masked REGION of an existing video, keeping everything outside it.

    The other half of the mask injector. `H3LatentPin` masks in TIME (pin the
    opening, generate the rest); this masks in SPACE (pin the surroundings,
    regenerate what the mask covers, on every frame). Pair it with a segmentation
    model — SAM, `ComfyUI-segment-anything-2`, anything producing a per-frame MASK —
    and reference anchors, and it becomes character replacement that does not depend
    on `[video editing]` being present in the open weights.

    WHY THIS SHOULD BEHAVE BETTER THAN THE TEMPORAL PIN
      Every join cut we ever traced sat at a TEMPORAL mask edge — the frame where
      pinned content stopped and generation began — and no amount of feathering
      moved it. A SPATIAL edge is a different problem: image inpainting deals with
      those routinely, and feathering genuinely helps, because the model can blend
      across a boundary it sees all at once rather than having to invent what comes
      after a wall in time.

    THE PART THAT NEEDS CARE — TEMPORAL DOWNSAMPLING
      The video VAE packs ~3.4 pixel frames into each latent frame, so a per-frame
      pixel mask cannot be sampled, it has to be UNIONED: if the subject occupies a
      pixel anywhere in the frames feeding a latent frame, that latent cell must be
      masked. Max-pooling does exactly that, which is why it is used here instead of
      interpolation. Under-masking leaves slivers of the original subject at the
      edges of fast motion; over-masking only costs a little extra regeneration, so
      `dilate` errs generous by default.

    Audio is pinned to the source by default — you are replacing a person, not the
    soundtrack. Turn `keep_audio` off to regenerate it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "source_images": ("IMAGE", {"tooltip": "The source video's frames."}),
            "mask": ("MASK", {"tooltip": "Per-frame subject mask. White = regenerate."}),
            "dilate": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1,
                               "tooltip": "Grow the mask in LATENT cells. Each cell is "
                                          "16 px, so 2 is ~32 px of margin. Too tight "
                                          "leaves slivers of the original subject."}),
            "feather": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Soften the boundary so the model can "
                                             "blend rather than butt up against a wall."}),
            "invert": ("BOOLEAN", {"default": False,
                                   "tooltip": "ON = keep the subject, regenerate the "
                                              "surroundings instead."}),
            "keep_audio": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "audio_vae": ("VAE",),
            "source_audio": ("AUDIO",),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Regenerate a masked region of an existing video while pinning "
                   "everything outside it. Feed a SAM mask and reference anchors to "
                   "replace a person without touching the scene.")

    def go(self, latent, vae, source_images, mask, dilate, feather, invert, keep_audio,
           audio_vae=None, source_audio=None):
        import comfy.nested_tensor
        import torch.nn.functional as Fn

        video, aud = _av(latent["samples"])
        lt, lh, lw = video.shape[2], video.shape[3], video.shape[4]

        # CONFORM RATHER THAN REFUSE. The latent's shape is set by the H3 node's
        # width/height/length; the source is whatever clip you loaded. Requiring the
        # two to be typed into agreement failed repeatedly in practice — the numbers
        # live on two different nodes and nothing keeps them in step. Spatially,
        # resampling the source to the latent's frame size is well defined, so do it
        # and say so. Frame COUNT is not resamplable (it is on the VAE's 17n+5 grid
        # and each latent frame spans several pixel frames), so that still errors.
        sw, sh = source_images.shape[2], source_images.shape[1]
        tw, th = lw * 16, lh * 16
        if (sw, sh) != (tw, th):
            ar_src, ar_dst = sw / max(1, sh), tw / max(1, th)
            if max(ar_src, ar_dst) / min(ar_src, ar_dst) > 1.05:
                logging.warning(
                    "H3MaskInpaint: source is %dx%d (%.2f:1) but the latent is %dx%d "
                    "(%.2f:1) — conforming will STRETCH the picture. Set the H3 node's "
                    "width/height to the source's aspect (or crop the source first).",
                    sw, sh, ar_src, tw, th, ar_dst)
            else:
                logging.info("H3MaskInpaint: conforming source %dx%d -> %dx%d to match "
                             "the latent", sw, sh, tw, th)
            src = Fn.interpolate(source_images.movedim(-1, 1), size=(th, tw),
                                 mode="bicubic", align_corners=False,
                                 antialias=True).clamp(0, 1).movedim(1, -1)
            mask = Fn.interpolate(
                (mask if mask.dim() == 3 else mask.unsqueeze(0)).unsqueeze(1),
                size=(th, tw), mode="nearest").squeeze(1)
        else:
            src = source_images

        z = vae.encode(src)                                 # [1,24,T,h,w]
        if z.shape[2] != lt:
            raise ValueError(
                f"source is {src.shape[0]} frame(s) -> {z.shape[2]} latent frames, but "
                f"the latent has {lt}. Frame count cannot be resampled: it sits on the "
                f"VAE's 17n+5 grid. Set the H3 node's `length` to match the source, or "
                f"put H3 Match Source Clip in front of it (it trims to a legal count).")

        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        m = m.float().unsqueeze(0).unsqueeze(0)             # [1,1,T,H,W]
        if invert:
            m = 1.0 - m

        # UNION, not resample: any pixel frame contributing to a latent frame counts
        m = Fn.adaptive_max_pool3d(m, (lt, lh, lw))

        if dilate > 0:
            k = dilate * 2 + 1
            m = Fn.max_pool3d(m, kernel_size=(1, k, k), stride=1,
                              padding=(0, dilate, dilate))
        if feather > 0:
            r = max(1, int(round(feather * 3)))
            k = r * 2 + 1
            m = Fn.avg_pool3d(m, kernel_size=(1, k, k), stride=1, padding=(0, r, r))
            m = m.clamp(0, 1)

        mask_v = m.expand(video.shape[0], video.shape[1], lt, lh, lw).contiguous()
        mask_v = mask_v.to(video.device, video.dtype)
        known_v = z.to(video.device, video.dtype)

        known_a, mask_a = aud, torch.ones_like(aud)
        if keep_audio and source_audio is not None and audio_vae is not None:
            wav = source_audio["waveform"]
            sr = source_audio["sample_rate"]
            vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if sr != vae_sr:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, vae_sr)
            za = audio_vae.encode(wav[:1].movedim(1, -1))
            za = za[..., :aud.shape[-1]]
            if za.shape[-1] < aud.shape[-1]:
                pad = aud.shape[-1] - za.shape[-1]
                za = torch.cat([za, za[..., -1:].expand(*za.shape[:-1], pad)], dim=-1)
            known_a = za.to(aud.device, aud.dtype)
            mask_a = torch.zeros_like(aud)

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((known_v, known_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        return (out,)


NODE_CLASS_MAPPINGS["H3MaskInpaint"] = H3MaskInpaint
NODE_DISPLAY_NAME_MAPPINGS["H3MaskInpaint"] = "H3 Mask Inpaint (region replace)"
