"""Teach ComfyUI's context windows how to window MiniMax-H3.

WHAT CORE DOES
  `comfy/context_windows.py` windows the model's forward pass over ONE long
  latent -- no join, no trim, no per-chunk VAE round-trip -- and the denoise mask
  is applied in `KSamplerX0Inpaint.__call__` on the FULL latent OUTSIDE the
  windowing, so a mask plus context windows composes into windowed masked V2V in
  a single sampler pass.

WHY H3 NEEDS ANYTHING AT ALL
  Core assumes every modality's temporal axis sits at the same tensor dim as the
  primary's. `LTXAV` satisfies that -- its audio time is dim 2. H3 does not: its
  video latent is `[B, 24, T, H/16, W/16]` (time at dim 2) but its audio latent
  is `[B, 32, 2, T]`, where dim 2 is the STEREO PAIR and time is the last axis.

  That assumption is baked into at least four sites, three of which this file
  used to monkey-patch. The fourth is inside a large core method, and vendoring
  it would break on every ComfyUI update -- two did in one evening on
  0.33.2 -> 0.34.0. So the fix moved upstream instead:

    An optional `context_modality_dim(modality_index, latent_shapes, dim)` hook
    on the model, defaulting to the primary dim, consulted wherever a modality's
    axis was previously assumed. Verified 2026-08-26 against master d8e7bbc:
    9 new tests fail unpatched and pass patched, and 24 configurations
    (3 schedules x 4 fuse methods x single/multi-modality) come out bit-identical
    before and after.

  This file now supplies only H3's SIDE of that contract. If core does not have
  the hook, the node says so plainly instead of failing mid-render.

WHAT WE KNOW THAT A GENERIC MAPPING CANNOT
  A proportional video->audio map assumes every latent frame covers the same
  number of pixel frames. H3's do not: the video VAE groups them (1, 4, 4, 4, 4),
  so every fifth latent frame covers ONE pixel frame and the rest cover four. A
  proportional split is off by up to three pixel frames, worst at the clip start
  -- latent 1 maps to audio 6 where it actually begins at audio 1. Same class of
  error as reducing a mask with equal buckets, a real bug here on 2026-08-19.
  This maps through the real frame grid.

STILL UNPROVEN
  Whether a windowed H3 render is CORRECT, as opposed to merely running.
  `payload["keyframes"]` and `payload["refs"]` carry absolute positions on the
  full clip's timeline, and what they do when the latent is windowed is under
  investigation. Expect wrong output before expecting a crash.
"""

import logging

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_HZ = 40

# H3 latent layouts. video [B, 24, T, H/16, W/16]; audio [B, 32, 2, T40].
VIDEO_TIME_DIM = 2
AUDIO_TIME_DIM = 3


def _pixel_frame_at(latent_index):
    """Pixel frame where a video latent frame begins, honouring (1,4,4,4,4)."""
    n = int(latent_index)
    full, rest = divmod(n, 5)
    return full * 17 + sum(FRAME_PER_TOKEN[:rest])


def _audio_span_for_latent(latent_index):
    """(start, end) audio latent indices covered by one video latent frame."""
    start_px = _pixel_frame_at(latent_index)
    span_px = FRAME_PER_TOKEN[int(latent_index) % 5]
    a0 = int(start_px * AUDIO_LATENT_HZ // FPS)
    a1 = int(-(-(start_px + span_px) * AUDIO_LATENT_HZ // FPS))   # ceil
    return a0, max(a1, a0 + 1)


def map_modalities(primary_indices, latent_shapes, dim):
    """Video window indices -> the audio indices covering the same real time."""
    result = [list(primary_indices)]
    if not latent_shapes or len(latent_shapes) < 2:
        return result
    audio_total = int(latent_shapes[1][AUDIO_TIME_DIM])
    seen, audio_indices = set(), []
    for v in primary_indices:
        a0, a1 = _audio_span_for_latent(v)
        for a in range(max(0, a0), min(a1, audio_total)):
            if a not in seen:
                seen.add(a)
                audio_indices.append(a)
    if not audio_indices:                       # never hand back an empty window
        audio_indices = [0]
    result.append(audio_indices)
    return result


def window_pixel_range(window):
    """(first_pixel_frame, last_pixel_frame_exclusive) covered by a video window."""
    idx = list(getattr(window, "index_list", []) or [])
    if not idx:
        return 0, 0
    first, last = min(idx), max(idx)
    start = _pixel_frame_at(first)
    end = _pixel_frame_at(last) + FRAME_PER_TOKEN[last % 5]
    return start, end


def rebase_keyframes(payload, window):
    """Payload with keyframes rebased to this window, or None if unchanged.

    THE BUG THIS FIXES
      `resolved_frame_index` is an absolute pixel frame on the FULL clip, and
      PackedLayout places it at `cursor + FRAME_RESCALE * idx`. But a windowed
      target video RESTARTS at `cursor` in every window. So a first-frame
      keyframe lands on frame 0 of EVERY window -- the clip snaps back to the
      start image at every seam -- and a later one floats past the window's span,
      ignored while still costing attention. Measured 2026-08-26.

      LTXAV solves the same problem with `compute_guide_overlap`: drop guides
      that do not overlap the window, rebase the rest to window-local positions.
      This is that, for H3.

      `cond_video_latents` / `cond_audio_latents` are rebuilt in core's order --
      keyframes FIRST, then refs (model_base.py:2186,2191) -- so dropping a
      keyframe must drop its latent from the same position.
    """
    kfs = payload.get("keyframes")
    if not kfs:
        return None

    start, end = window_pixel_range(window)
    if end <= start:
        return None

    kept = []
    for kf in kfs:
        idx = int(kf.get("resolved_frame_index", 0))
        if start <= idx < end:
            local = dict(kf)
            local["resolved_frame_index"] = idx - start
            kept.append(local)

    if len(kept) == len(kfs) and all(
            k["resolved_frame_index"] == kf.get("resolved_frame_index")
            for k, kf in zip(kept, kfs)):
        return None                       # window covers the clip; nothing to do

    out = dict(payload)
    refs = out.get("refs") or []
    if kept:
        out["keyframes"] = kept
    else:
        out.pop("keyframes", None)

    vid = [k["latent"] for k in kept if k.get("latent") is not None]
    vid += [r["latent"] for r in refs if "latent" in r]
    aud = [k["audio_latent"] for k in kept if k.get("audio_latent") is not None]
    aud += [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    if vid:
        out["cond_video_latents"] = vid
    else:
        out.pop("cond_video_latents", None)
    if aud:
        out["cond_audio_latents"] = aud
    else:
        out.pop("cond_audio_latents", None)

    # the layout is rebuilt per window off its signature, so dropping a stale one
    # is not required -- but keeping a layout built for different keyframes is a
    # trap waiting for the signature check to be relaxed
    out.pop("layout", None)
    return out


def core_has_modality_dim_hook():
    """True when this ComfyUI consults `context_modality_dim`.

    Reads the source FILE rather than importing the module. Two reasons: the
    hook lives on the MODEL, so its presence there says nothing about whether
    core calls it; and importing comfy.context_windows pulls in model_management
    and initialises CUDA, which fails outright on a machine with no GPU and
    would be caught as "no support" -- the wrong answer for the wrong reason.
    `find_spec` locates the file without executing it.
    """
    try:
        import importlib.util
        import sys
        mod = sys.modules.get("comfy.context_windows")
        path = getattr(mod, "__file__", None) if mod is not None else None
        if not path:
            spec = importlib.util.find_spec("comfy.context_windows")
            path = getattr(spec, "origin", None) if spec else None
        if not path:
            return False
        with open(path, encoding="utf-8") as fh:
            return "context_modality_dim" in fh.read()
    except Exception:
        return False


def patch_h3_context_windows():
    """Install H3's side of the modality-dim contract. -> (usable, message)."""
    try:
        import torch
        import comfy.model_base as MB
    except Exception as exc:
        return False, f"comfy.model_base not importable: {type(exc).__name__}"

    H3 = getattr(MB, "MiniMaxH3", None)
    if H3 is None:
        return False, "MiniMaxH3 not found in comfy.model_base."

    if not getattr(H3, "_h3_windowing_patched", False):
        def _modality_dim(self, modality_index, latent_shapes, dim):
            # audio is [B, 32, 2, T]: dim 2 is the stereo pair, time is last
            if modality_index >= 1 and latent_shapes:
                return len(latent_shapes[modality_index]) - 1
            return dim

        def _map(self, primary_indices, latent_shapes, dim):
            return map_modalities(primary_indices, latent_shapes, dim)

        def _resize(self, cond_key, cond_value, window, x_in, device,
                    retain_index_list=[]):
            cond = getattr(cond_value, "cond", None)
            if cond_key == "audio_denoise_mask" and isinstance(cond, torch.Tensor):
                aw = (getattr(window, "modality_windows", None) or {}).get(1)
                if aw is not None:
                    return cond_value._copy_with(
                        aw.get_tensor(cond, device, dim=AUDIO_TIME_DIM))
            # keyframes carry ABSOLUTE clip positions; rebase them per window
            if cond_key == "minimax_payload" and isinstance(cond, dict):
                fixed = rebase_keyframes(cond, window)
                if fixed is not None:
                    return cond_value._copy_with(fixed)
                return None

            if cond_key == "denoise_mask" and isinstance(cond, torch.Tensor):
                return cond_value._copy_with(
                    window.get_tensor(cond, device,
                                      retain_index_list=retain_index_list))
            return MB.BaseModel.resize_cond_for_context_window(
                self, cond_key, cond_value, window, x_in, device,
                retain_index_list=retain_index_list)

        H3.context_modality_dim = _modality_dim
        H3.map_context_window_to_modalities = _map
        H3.resize_cond_for_context_window = _resize
        H3._h3_windowing_patched = True

    if core_has_modality_dim_hook():
        return True, (
            "H3 context windowing ENABLED.\n"
            "  core supports context_modality_dim; H3 hooks installed:\n"
            "    audio time resolved to dim 3, not the stereo pair at dim 2\n"
            "    video->audio mapped on the real (1,4,4,4,4) frame grid, not\n"
            "    proportionally — proportional is off by up to 3 pixel frames,\n"
            "    worst at the clip start\n"
            "  Now wire Context Windows (Manual) into the sampler, with dim = 2.\n"
            "\n"
            "  Bounds ATTENTION cost, not memory — the whole latent stays\n"
            "  resident. UNPROVEN: whether keyframes and reference blocks, which\n"
            "  carry absolute timeline positions, survive being windowed. Expect\n"
            "  wrong output before a crash; check the seams.")

    return False, (
        "H3 hooks installed, but THIS ComfyUI CANNOT USE THEM.\n"
        "  comfy/context_windows.py has no context_modality_dim support, so it\n"
        "  still assumes every modality's time axis sits at the primary's dim.\n"
        "  H3's audio is [B, 32, 2, T] — dim 2 is the stereo pair. Windowing will\n"
        "  fail, most likely as\n"
        "    'size of tensor a (2) must match the size of tensor b (N) at dim 2'\n"
        "\n"
        "  The fix is a small upstream ComfyUI patch, verified 2026-08-26. Until\n"
        "  it lands, use H3 Chunk Plan with H3 Chunk Open/Close — separate passes\n"
        "  need no per-modality windowing at all.")


class H3EnableContextWindows:
    """Make ComfyUI's context windows work on MiniMax-H3.

    Pass the model through once before the sampler, then use core's
    `Context Windows (Manual)` with `dim = 2`. The MODEL passes through
    unchanged; this only installs H3's side of the modality-dim contract.

    Read `info`: it says whether this ComfyUI supports the contract, and if not,
    what to use instead — rather than letting a render find out.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    EXPERIMENTAL = True
    DESCRIPTION = ("Install MiniMax-H3's context-window hooks so core can window "
                   "it. Reports whether this ComfyUI supports them.")

    def go(self, model):
        usable, text = patch_h3_context_windows()
        logging.info("H3EnableContextWindows: %s", text.splitlines()[0])
        return {"ui": {"h3char": [text]}, "result": (model, text)}


class H3ContextWindows:
    """Context windows for H3, in FRAMES, phase-aligned, on the right axis.

    Core's `Context Windows (Manual)` is usable for H3 but has three ways to get
    it silently wrong, and this node closes all three:

      `dim` DEFAULTS TO 0, the batch axis. H3's video temporal axis is 2. Wrong
      dim does not error, it windows the wrong thing.

      Its lengths are in LATENT frames. 27 latent frames is 90 pixel frames, and
      the conversion is not a constant ratio -- the VAE groups pixel frames
      (1,4,4,4,4), so the mapping is 17 pixel frames per 5 latent frames.

      PHASE. `_video_t_spans` indexes FRAME_PER_TOKEN[k % 5] from k=0, so
      PackedLayout assumes every window starts on a VAE chunk boundary. A window
      starting at a latent index not divisible by 5 gets a wrong intra-window
      frame grid -- measured 2026-08-26 at 5 rope-t units, 3 pixel frames,
      0.125 s, plus up to 0.09 s of video/audio start offset. Both the window
      length AND the stride must be multiples of 5 latent frames, which is
      multiples of 17 PIXEL frames. This node snaps them and says so.

    What windowing buys over chaining: there is ONE latent and one denoising
    process, so no re-encode contrast climb, no per-link identity re-anchoring,
    and no prefix-then-fresh-decision seam. What it does not buy is memory --
    the whole latent stays resident -- and every window is positioned at the clip
    origin, so continuity across a seam comes only from overlap blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Run it through H3 Enable Context "
                                           "Windows first."}),
            "window_frames": ("INT", {"default": 85, "min": 17, "max": 3600,
                              "step": 17,
                              "tooltip": "Window size in PIXEL frames. Snapped to "
                                         "a multiple of 17, which is what keeps "
                                         "each window on a VAE chunk boundary. "
                                         "85 frames = 25 latent frames."}),
            "overlap_frames": ("INT", {"default": 17, "min": 0, "max": 3600,
                               "step": 17,
                               "tooltip": "Also snapped to a multiple of 17, so "
                                          "the STRIDE stays aligned too. Overlap "
                                          "is the only thing carrying continuity "
                                          "across a seam."}),
        }, "optional": {
            "schedule": (["standard_static", "standard_uniform", "looped_uniform",
                          "batched"], {"default": "standard_static"}),
            "fuse_method": (["pyramid", "flat", "overlap-linear", "relative"],
                            {"default": "pyramid"}),
            "freenoise": ("BOOLEAN", {"default": False,
                          "tooltip": "Noise shuffling to improve blending. "
                                     "UNTESTED on H3."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    EXPERIMENTAL = True
    DESCRIPTION = ("Context windows sized in frames, snapped to H3's VAE chunk "
                   "boundary, on the correct temporal axis.")

    def go(self, model, window_frames, overlap_frames, schedule="standard_static",
           fuse_method="pyramid", freenoise=False):
        notes = []

        wf = max(17, int(round(window_frames / 17.0)) * 17)
        of = max(0, int(round(overlap_frames / 17.0)) * 17)
        if of >= wf:
            of = wf - 17
            notes.append(f"overlap must be smaller than the window — cut to {of}")
        if wf != int(window_frames):
            notes.append(f"window {int(window_frames)} -> {wf} frames (multiple of 17)")
        if of != int(overlap_frames):
            notes.append(f"overlap {int(overlap_frames)} -> {of} frames (multiple of 17)")

        w_lat = wf // 17 * 5
        o_lat = of // 17 * 5

        # Look the node up by its NODE ID, not by a Python class name. The class
        # is `ContextWindowsManualNode` while the id is `ContextWindowsManual`,
        # and importing the guessed name failed outright. The id is the stable
        # contract -- it is what saved workflows store -- so resolve through the
        # registry and fall back to the module only if that misses.
        CWM = None
        try:
            import nodes as comfy_nodes
            CWM = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(
                "ContextWindowsManual")
        except Exception:
            pass
        if CWM is None:
            try:
                import comfy_extras.nodes_context_windows as M
                CWM = (getattr(M, "ContextWindowsManualNode", None)
                       or getattr(M, "ContextWindowsManual", None))
            except Exception as exc:
                CWM = None
                notes.append(f"import failed: {type(exc).__name__}: {exc}")
        if CWM is None:
            msg = ("Context Windows node not found. This ComfyUI has no\n"
                   "  ContextWindowsManual — windowing is unavailable here.")
            return {"ui": {"h3char": [msg]}, "result": (model, msg)}

        out = CWM.execute(model=model, context_length=w_lat, context_overlap=o_lat,
                          context_schedule=schedule, context_stride=1,
                          closed_loop=False, fuse_method=fuse_method,
                          dim=VIDEO_TIME_DIM, freenoise=freenoise,
                          cond_retain_index_list=[], split_conds_to_windows=False,
                          latent_retain_index_list=[], causal_window_fix=True)
        patched = out.result[0] if hasattr(out, "result") else (
            out[0] if isinstance(out, (list, tuple)) else out)

        text = "\n".join([
            f"H3 context windows: {wf} frames ({w_lat} latent), overlap {of} "
            f"({o_lat} latent)",
            f"  stride {wf - of} frames ({w_lat - o_lat} latent) — both multiples "
            f"of 5 latent, so every window starts on a VAE chunk boundary",
            f"  dim {VIDEO_TIME_DIM} (H3's video temporal axis; core defaults to 0)",
            f"  schedule {schedule}, fuse {fuse_method}",
            "",
            "  bounds ATTENTION, not memory — the whole latent stays resident",
        ] + [f"  NOTE {n}" for n in notes])
        return {"ui": {"h3char": [text]}, "result": (patched, text)}


NODE_CLASS_MAPPINGS = {"H3EnableContextWindows": H3EnableContextWindows,
                       "H3ContextWindows": H3ContextWindows}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EnableContextWindows": "H3 Enable Context Windows",
    "H3ContextWindows": "H3 Context Windows"}
