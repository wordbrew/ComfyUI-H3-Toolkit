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


# The mode rides on the MODEL, not on a module global.
#
# It was a module global, set as a side effect of H3ContextWindows running --
# and ComfyUI caches node outputs, so on any queue where nothing upstream
# changed the node never ran, the flag stayed at its import default of False,
# and absolute positioning turned itself off silently. First render after a
# restart was right and every one after it was wrong. Measured 2026-08-28.
#
# An attribute on the BaseModel survives the cache because the cached
# ModelPatcher wraps the same model object, and the hook is a bound method on
# that model, so `self` is exactly where to read it.
ABSOLUTE_FLAG = "_h3_absolute_window_positions"


def frames_for_latent(latent_t):
    """Pixel frames for a latent count on the 5n+2 grid. Inverse of video_latent_t."""
    lt = int(latent_t)
    return 5 if lt <= 2 else ((lt - 2) // 5) * 17 + 5


def window_schedule(latent_len, w_lat, o_lat):
    """The windows core will actually build. -> [(start, end_exclusive), ...]

    A REPLICA of comfy.context_windows.create_windows_static_standard, kept here
    so the plan can be read without importing comfy (which initialises CUDA).
    The clamp is the part worth predicting: when the last window would run past
    the end it is pulled BACK to fit, which silently makes its overlap with the
    previous one bigger than the one you asked for -- a 141 window with a 39
    overlap on a 192-frame clip ends up sharing 27 latent frames, not 12.
    """
    delta = max(1, w_lat - o_lat)
    out = []
    for start in range(0, latent_len, delta):
        if start + w_lat >= latent_len:
            final = max(0, latent_len - w_lat)
            out.append((final, final + w_lat))
            break
        out.append((start, start + w_lat))
    return out or [(0, w_lat)]


def core_takes_window_start():
    """True when this ComfyUI's PackedLayout accepts a window_start.

    Read from the SOURCE rather than imported: importing comfy.ldm.minimax.model
    initialises CUDA, which a node listing must not do.
    """
    try:
        from importlib.util import find_spec
        spec = find_spec("comfy.ldm.minimax.model")
        if not spec or not spec.origin:
            return False
        with open(spec.origin, encoding="utf-8") as fh:
            return "window_start" in fh.read()
    except Exception:
        return False


def window_start_pixel(window):
    """Clip PIXEL frame where this window begins, honouring (1,4,4,4,4)."""
    idxs = getattr(window, "index_list", None)
    if not idxs:
        return 0
    return _pixel_frame_at(int(min(idxs)))


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
            if cond_key == "minimax_payload" and isinstance(cond, dict):
                # keyframes carry ABSOLUTE clip positions; rebase them per window
                fixed = rebase_keyframes(cond, window)
                if getattr(self, ABSOLUTE_FLAG, False):
                    # The target now sits at its real place on the clip's
                    # timeline, so keyframe indices are already in that frame of
                    # reference and rebasing them would move them twice. Keep
                    # the DROP of keyframes outside the window -- they cost
                    # attention and describe frames this window does not hold --
                    # and put the absolute index back.
                    base = dict(fixed if fixed is not None else cond)
                    kept = base.get("keyframes")
                    if kept:
                        start = window_start_pixel(window)
                        base["keyframes"] = [
                            {**k, "resolved_frame_index":
                                int(k.get("resolved_frame_index", 0)) + start}
                            for k in kept]
                    start = window_start_pixel(window)
                    base["window_start_frames"] = start
                    return cond_value._copy_with(base)
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
            # min 39 with step 17 IS the legal-run grid (17n+5) from the
            # smallest useful run up. It used to be min 17 step 17, which walks
            # the MULTIPLES of 17 -- 17, 34, 51, 68, 85 -- and not one of those
            # is a legal run, so the arrows could only ever produce a value the
            # node then silently corrected. The default was 85 for the same
            # reason: 85 is 17x5 and snaps to 90.
            "window_frames": ("INT", {"default": 90, "min": 39, "max": 3600,
                              "step": 17,
                              "tooltip": "Window size in PIXEL frames, a legal "
                                         "run (17n+5). 90 frames = 27 latent "
                                         "frames and lands on both clocks."}),
            "overlap_frames": ("INT", {"default": 39, "min": 5, "max": 3600,
                               "step": 17,
                               "tooltip": "Also a legal run, so the STRIDE stays "
                                          "aligned too. With a 90 window, 39 "
                                          "gives a stride of 51 — divisible by 3, "
                                          "so every window also starts on an "
                                          "exact 40 Hz audio tick. 22 gives 68, "
                                          "which does not. Overlap is the only "
                                          "thing carrying continuity across a "
                                          "seam."}),
        }, "optional": {
            "schedule": (["standard_static", "standard_uniform", "looped_uniform",
                          "batched"], {"default": "standard_static"}),
            "fuse_method": (["pyramid", "flat", "overlap-linear", "relative"],
                            {"default": "pyramid"}),
            "freenoise": ("BOOLEAN", {"default": False,
                          "tooltip": "Noise shuffling to improve blending. "
                                     "UNTESTED on H3."}),
            # APPENDED. Core defaults this True and it was hardcoded True here,
            # untested. Measured 2026-08-28: it makes every window after the
            # first 28 latent frames where window 0 is 27, and 28 is NOT on H3's
            # 5n+2 latent grid — the VAE groups pixel frames (1,4,4,4,4), so 27
            # latent is exactly 90 pixel frames and 28 is 94, which is not a
            # legal run. Written for causal-VAE models; unproven on this one.
            "causal_window_fix": ("BOOLEAN", {"default": True,
                                  "tooltip": "Core adds one 'causal fix' frame to "
                                             "every window after the first. On H3 "
                                             "that lands them off the VAE's 5n+2 "
                                             "latent grid — 27 latent is 90 pixel "
                                             "frames, 28 is 94 and is not a legal "
                                             "run. Turn it OFF to keep every "
                                             "window on the grid."}),
            # APPENDED. The measurement this exists to settle: every window's
            # target was placed at `cursor`, which does not depend on the window,
            # so each one rendered the OPENING of the shot and the overlaps
            # crossfaded between openings. Confirmed at runtime 2026-08-28 --
            # 167 layout builds, one cond_t. Needs the matching core patch
            # (patches/h3-window-absolute-positions.patch).
            # APPENDED. The measurement this exists to settle: every window's
            # target was placed at `cursor`, which does not depend on the window,
            # so each one rendered the OPENING of the shot and the overlaps
            # crossfaded between openings. Confirmed at runtime 2026-08-28 --
            # 167 layout builds, one cond_t. Needs the matching core patch
            # (patches/h3-window-absolute-positions.patch).
            "absolute_window_positions": ("BOOLEAN", {"default": False,
                                          "tooltip": "Give each window its real "
                                                     "position on the clip's "
                                                     "timeline instead of the "
                                                     "clip origin, so a later "
                                                     "window reads as 'this clip, "
                                                     "N frames in' rather than "
                                                     "'another clip'. Needs the "
                                                     "core patch; says so if it "
                                                     "is missing."}),
            # APPENDED. Core can already hand each window a DIFFERENT
            # conditioning, chosen by where the window's centre falls in the
            # clip: region = int(center_ratio * len(conds)), with center_ratio =
            # (first + last) / (2 * total). Combine N prompts with Conditioning
            # (Combine) and windowing becomes long-form prompting -- the thing
            # windowing otherwise cannot do.
            #
            # UNPROVEN ON H3, and two things are unknown. Each conditioning
            # carries its OWN minimax_payload with its own PackedLayout built
            # from its own text_len, and whether the split survives that is the
            # same machinery the keyframe rebasing already patches. And the
            # region map is arithmetic on window centres, not on where your
            # clauses change: three prompts across nine windows put the
            # boundaries wherever they land.
            #
            # What IS known: overlaps still BLEND. A prompt change is a crossfade
            # the width of the overlap, not a cut. For a description that evolves
            # that is a feature; for dialogue it is not, because two speech
            # signals averaged in latent space do not give clean speech.
            "split_conds_to_windows": ("BOOLEAN", {"default": False,
                                       "tooltip": "Give each window its own "
                                                  "conditioning, picked by where "
                                                  "the window's CENTRE falls in "
                                                  "the clip. Combine N prompts "
                                                  "with Conditioning (Combine) "
                                                  "first. Overlaps blend, so a "
                                                  "prompt change is a crossfade "
                                                  "the width of the overlap — "
                                                  "good for an evolving shot, "
                                                  "bad for dialogue."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    EXPERIMENTAL = True
    DESCRIPTION = ("Context windows sized in frames, snapped to H3's VAE chunk "
                   "boundary, on the correct temporal axis.")

    def go(self, model, window_frames, overlap_frames, schedule="standard_static",
           fuse_method="pyramid", freenoise=False, causal_window_fix=True,
           absolute_window_positions=False, split_conds_to_windows=False):
        from .timing import snap_run, video_latent_t

        notes = []

        # A UNIFORM schedule moves every window on every step: `pad = round(
        # num_frames * ordered_halving(step))`, so step 0 puts them at 0/15/30
        # and step 1 at 28/43/... On this model that is actively wrong. No latent
        # frame is rendered by a window in a consistent position, the fuse
        # weights are computed for wherever it happened to land, and with
        # absolute positions each frame is told a different time on every step.
        # Measured 2026-08-28: core logged 3 windows on one step and 4 on the
        # next, the second set starting at 28 — from a node whose widget read
        # standard_static.
        if schedule != "standard_static":
            notes.append(f"SCHEDULE IS {schedule.upper()}, NOT standard_static. "
                         f"Uniform schedules move every window on every step, so "
                         f"no frame is rendered by a window in a consistent "
                         f"place. Set schedule to standard_static unless you are "
                         f"deliberately testing this.")

        absolute = bool(absolute_window_positions)
        if absolute and not core_takes_window_start():
            notes.append("absolute_window_positions asked for, but this "
                         "ComfyUI's PackedLayout does not accept window_start — "
                         "apply patches/h3-window-absolute-positions.patch. "
                         "Running with origin-positioned windows.")
            absolute = False

        # BOTH must be legal runs (17n+5), not merely multiples of 17.
        # A clip's latent length is always 5k+2, so the FINAL window begins at
        # latent_t - window_len, which lands on a multiple of 5 only when the
        # window length is ALSO 5k+2 -- that is, only when the window is itself
        # a legal run. An earlier version snapped to multiples of 17 and
        # converted with frames//17*5, which is not the real map: it asked for
        # 25-latent windows while reporting 85 frames (85 frames is 22 latent),
        # and on an 87-latent clip the last window started at 62, off-phase.
        wf = snap_run(max(39, int(window_frames)))
        of = snap_run(max(5, int(overlap_frames))) if int(overlap_frames) > 0 else 0
        if of >= wf:
            of = snap_run(max(5, wf // 3))
            notes.append(f"overlap must be smaller than the window - cut to {of}")
        if wf != int(window_frames):
            notes.append(f"window {int(window_frames)} -> {wf} frames "
                         f"(nearest legal run, 17n+5)")
        if of != int(overlap_frames):
            notes.append(f"overlap {int(overlap_frames)} -> {of} frames "
                         f"(nearest legal run)")

        w_lat = video_latent_t(wf)
        o_lat = video_latent_t(of) if of > 0 else 0
        stride = w_lat - o_lat
        if stride % 5:
            notes.append(f"stride {stride} latent is not a multiple of 5 - "
                         f"intermediate windows will be off-phase")

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
                          cond_retain_index_list=[],
                          split_conds_to_windows=bool(split_conds_to_windows),
                          latent_retain_index_list=[],
                          causal_window_fix=bool(causal_window_fix))
        patched = out.result[0] if hasattr(out, "result") else (
            out[0] if isinstance(out, (list, tuple)) else out)
        base = getattr(patched, "model", None)
        if base is not None:
            setattr(base, ABSOLUTE_FLAG, absolute)

        # One line per RUN saying what was actually installed. The afternoon this
        # cost was spent on a widget that read standard_static while the handler
        # ran uniform, and on a mode that turned itself off when the node was
        # cached -- both invisible from the outside, both one line to catch.
        logging.info("H3 context windows: %s frames (%s latent), overlap %s (%s), "
                     "stride %s | schedule %s | fuse %s | causal_fix %s | "
                     "absolute positions %s | split conds %s",
                     wf, w_lat, of, o_lat, wf - of, schedule, fuse_method,
                     causal_window_fix, absolute, split_conds_to_windows)

        text = "\n".join([
            f"H3 context windows: {wf} frames ({w_lat} latent), overlap {of} "
            f"({o_lat} latent)",
            f"  stride {wf - of} frames ({stride} latent)",
            f"  both are legal runs (17n+5), so every window — including the "
            f"clamped last one — starts on a VAE chunk boundary",
            f"  dim {VIDEO_TIME_DIM} (H3's video temporal axis; core defaults to 0)",
            f"  schedule {schedule}, fuse {fuse_method}"
            + ("" if schedule == "standard_static" else "   <-- NOT static"),
            (f"  absolute positions ON — each window is placed at its real "
             f"frame on the clip timeline"
             if absolute else
             f"  absolute positions off — every window is placed at the clip "
             f"origin, so each renders the opening of the shot"),
            (f"  causal_window_fix ON — every window after the first gets "
             f"{w_lat + 1} latent frames, not {w_lat}, which is OFF H3's 5n+2 grid"
             if causal_window_fix else
             f"  causal_window_fix off — every window is {w_lat} latent frames"),
            "",
            "  bounds ATTENTION, not memory — the whole latent stays resident",
        ] + [f"  NOTE {n}" for n in notes])
        return {"ui": {"h3char": [text]}, "result": (patched, text)}



class H3WindowPlan:
    """What the window schedule will actually be, before you spend the render.

    Windowing has three numbers that have to agree and none of them tell you so:
    the clip length, the window and the overlap. Get them wrong and it still
    runs -- it just runs something other than what you asked for. Two ways that
    happens, both hit while measuring this:

      THE LAST WINDOW CLAMPS. When it would run past the end it is pulled BACK
      to fit, so its overlap with the previous one is bigger than the one you
      set. A 141 window with a 39 overlap on a 192-frame clip shares 27 latent
      frames, not 12 -- a gentler test than the settings imply, and not
      comparable to a run where the numbers divide.

      THE STRIDE LEAVES THE AUDIO GRID. Every window has to start on an exact
      40 Hz tick, which means the stride must divide by 3. At a 90 window that
      leaves 39 as the only legal overlap: 5, 22 and 56 all give strides of 85,
      68 and 34, and none of them divide.

    `length` is an output so the number the schedule was planned around is the
    number the conditioning node renders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "window_frames": ("INT", {"default": 141, "min": 39, "max": 3600,
                              "step": 17,
                              "tooltip": "Same value as on H3 Context Windows."}),
            "overlap_frames": ("INT", {"default": 39, "min": 5, "max": 3600,
                               "step": 17,
                               "tooltip": "Same value as on H3 Context Windows. "
                                          "It is the ONLY channel between "
                                          "windows, so it is the last thing to "
                                          "cut for speed."}),
            "mode": (["length -> windows", "windows -> length"],
                     {"default": "windows -> length",
                      "tooltip": "Either check a length you already have, or ask "
                                 "for the length that gives N clean windows."}),
        }, "optional": {
            "total_frames": ("INT", {"default": 345, "min": 5, "max": 36000,
                             "step": 1,
                             "tooltip": "For `length -> windows`."}),
            "windows": ("INT", {"default": 3, "min": 1, "max": 64,
                        "tooltip": "For `windows -> length`."}),
        }}

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("length", "window_count", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Work out the clip length that gives clean context windows, "
                   "or check the schedule a length will actually produce.")

    def go(self, window_frames, overlap_frames, mode, total_frames=345,
           windows=3):
        from .timing import snap_run, video_latent_t

        notes = []
        wf = snap_run(max(39, int(window_frames)))
        of = snap_run(max(5, int(overlap_frames)))
        if of >= wf:
            of = snap_run(max(5, wf // 3))
            notes.append(f"overlap must be smaller than the window — cut to {of}")
        for asked, got, what in ((int(window_frames), wf, "window_frames"),
                                 (int(overlap_frames), of, "overlap_frames")):
            if asked != got:
                notes.append(f"{what} {asked} is not a legal run — using {got}")

        w_lat, o_lat = video_latent_t(wf), video_latent_t(of)
        delta = w_lat - o_lat
        stride = frames_for_latent(delta + 2) - 5   # pixel frames per step
        if delta % 5:
            notes.append(f"latent stride {delta} is not a multiple of 5 — "
                         f"windows land off the VAE's frame grid")
        if stride % 3:
            notes.append(f"stride {stride} frames does not divide by 3, so "
                         f"windows start on fractional 40 Hz audio ticks. At a "
                         f"{wf} window the overlaps that do are: "
                         + (", ".join(str(o) for o in range(5, wf, 17)
                                      if (wf - o) % 3 == 0) or "none"))

        if mode == "windows -> length":
            n = max(1, int(windows))
            length = frames_for_latent(w_lat + delta * (n - 1))
        else:
            length = snap_run(max(5, int(total_frames)))
            if length != int(total_frames):
                notes.append(f"total_frames {int(total_frames)} is not a legal "
                             f"run — using {length}")

        lat = video_latent_t(length)
        sched = window_schedule(lat, w_lat, o_lat)
        rows, clamped = [], 0
        for i, (a, b) in enumerate(sched):
            px_a, px_b = _pixel_frame_at(a), _pixel_frame_at(b - 1) + 1
            share = ""
            if i:
                prev_end = sched[i - 1][1]
                ov = max(0, prev_end - a)
                share = f"  overlap {ov} latent"
                if ov != o_lat:
                    share += f" (asked {o_lat} — clamped)"
                    clamped += 1
            rows.append(f"  {i + 1:02d}  latent {a:4d}-{b - 1:<4d}  frames "
                        f"{px_a:5d}-{px_b:<5d}{share}")

        head = [f"{len(sched)} window(s) over {length} frames ({lat} latent)",
                f"  window {wf}f ({w_lat} latent), overlap {of}f ({o_lat}), "
                f"stride {stride}f ({delta} latent)"
                f"{'' if stride % 3 else '  — on both clocks'}",
                f"  {len(sched)} forward passes per step; the whole latent stays "
                f"resident either way"]
        if clamped:
            head.append(f"  {clamped} window(s) CLAMPED: the last one is pulled "
                        f"back to fit, so it shares more than you asked for. Use "
                        f"`windows -> length` for a length that divides.")
        text = "\n".join(head + rows + [f"  NOTE {n}" for n in notes])
        logging.info("H3WindowPlan: %s", head[0])
        return {"ui": {"h3char": [text]},
                "result": (length, len(sched), text)}


NODE_CLASS_MAPPINGS = {"H3EnableContextWindows": H3EnableContextWindows,
                       "H3ContextWindows": H3ContextWindows,
                       "H3WindowPlan": H3WindowPlan}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EnableContextWindows": "H3 Enable Context Windows",
    "H3ContextWindows": "H3 Context Windows",
    "H3WindowPlan": "H3 Window Plan (length <-> windows)"}
