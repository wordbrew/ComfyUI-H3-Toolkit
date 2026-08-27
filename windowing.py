"""Teach ComfyUI's context windows how to window MiniMax-H3.

WHAT CORE ALREADY DOES
  `comfy/context_windows.py` windows the model's forward pass over ONE long
  latent -- no join, no trim, no per-chunk VAE round-trip. It is the
  master-latent architecture, natively, and the denoise mask is applied in
  `KSamplerX0Inpaint.__call__` on the FULL latent OUTSIDE the windowing
  (`comfy/samplers.py`), so `SetLatentNoiseMask` + context windows composes into
  windowed masked V2V in a single sampler pass.

WHY H3 CANNOT USE IT UNPATCHED
  `LTXAV` implements `map_context_window_to_modalities` and
  `resize_cond_for_context_window`, which is how a joint audio+video model gets
  its masks sliced per window. `MiniMaxH3` inherits `BaseModel` and implements
  NEITHER, even though it has the whole masked-denoise machinery.

  Two further obstacles are not missing methods at all. Core's windowing assumes
  every modality's temporal axis sits at the SAME dim as the primary's, and H3
  breaks that in two separate places:

    `WindowingState.prepare_window` computes a modality's length as
    `self.latents[i].shape[self.dim]` and builds its window with `dim=self.dim`.
    For H3's `[B, 32, 2, T]` audio that reads the STEREO axis, not the length.

    `IndexListContextHandler.combine_context_window_results` builds ONE weights
    tensor from the primary dim and multiplies EVERY modality's output by it.
    That killed a real render with
      "size of tensor a (2) must match the size of tensor b (150) at dim 2"
    -- 2 being the stereo axis and 150 the audio length for a 90-frame window.

  LTXAV escapes both only because its audio time genuinely sits at dim 2. This
  file patches all three points. Scope was misjudged twice on the way here: the
  research said "two methods", reading said "three points", and the first render
  found a fourth. There may be more.

WHAT WE KNOW THAT CORE DOES NOT
  LTXAV maps video indices to audio PROPORTIONALLY. For H3 that is wrong. The
  video VAE groups pixel frames (1, 4, 4, 4, 4): every fifth latent frame covers
  ONE pixel frame and the rest cover four. A proportional map is off by up to
  three pixel frames, worst at the start, and puts a window's audio boundary in
  the wrong place -- the same class of error as reducing a mask with equal
  buckets, which was a real bug here on 2026-08-19. This maps through the actual
  frame grid instead.

UNTESTED. Written against ComfyUI master's context-window API and reviewed, but
it has not been run -- there is no torch in the environment it was written in.
Treat the first render as the test.
"""

import logging

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_HZ = 40

# H3 latent layouts. video [B, 24, T, H/16, W/16]; audio [B, 32, 2, T40].
VIDEO_TIME_DIM = 2
AUDIO_TIME_DIM = 3

_PATCHED = False


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
    """Video window indices -> the audio indices covering the same time.

    Replaces LTXAV's proportional mapping, which assumes every latent frame
    covers the same number of pixel frames. H3's do not.
    """
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


def patch_h3_context_windows():
    """Idempotent. Returns True if H3 can now be windowed, False if not."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import torch
        import comfy.model_base as MB
        import comfy.context_windows as CW
        import comfy.patcher_extension
    except Exception:
        return False

    H3 = getattr(MB, "MiniMaxH3", None)
    if H3 is None:
        return False
    if getattr(H3, "_h3_windowing_patched", False):
        _PATCHED = True
        return True

    # ---- 1. the two model hooks LTXAV has and H3 does not ------------------ #

    def _map(self, primary_indices, latent_shapes, dim):
        return map_modalities(primary_indices, latent_shapes, dim)

    def _resize(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        cond = getattr(cond_value, "cond", None)

        # audio mask [B, 1, 2, T] -- time is the LAST axis, not `window.dim`
        if cond_key == "audio_denoise_mask" and isinstance(cond, torch.Tensor):
            aw = (window.modality_windows or {}).get(1) if getattr(
                window, "modality_windows", None) else None
            if aw is not None:
                return cond_value._copy_with(
                    aw.get_tensor(cond, device, dim=AUDIO_TIME_DIM))

        # video mask [B, 1, T, H, W] -- ordinary temporal slice
        if cond_key == "denoise_mask" and isinstance(cond, torch.Tensor):
            return cond_value._copy_with(
                window.get_tensor(cond, device, retain_index_list=retain_index_list))

        return MB.BaseModel.resize_cond_for_context_window(
            self, cond_key, cond_value, window, x_in, device,
            retain_index_list=retain_index_list)

    H3.map_context_window_to_modalities = _map
    H3.resize_cond_for_context_window = _resize
    H3._h3_windowing_patched = True

    # ---- 2. tell the window builder where H3's audio time actually is ------ #
    # Core reads a modality's length as latents[i].shape[self.dim], which is the
    # VIDEO temporal axis. For H3's [B, 32, 2, T] audio that is the stereo axis.
    state = getattr(CW, "WindowingState", None)
    if state is not None and not getattr(state, "_h3_windowing_patched", False):
        original = state.prepare_window

        def prepare_window(self, window, model):
            out = original(self, window, model)
            if not getattr(self, "is_multimodal", False):
                return out
            if not isinstance(model, H3):
                return out
            mw = getattr(out, "modality_windows", None)
            if not mw:
                return out
            # rebuild the audio window on the correct axis
            for idx, sub in list(mw.items()):
                if idx < 1 or idx >= len(self.latents):
                    continue
                total = int(self.latents[idx].shape[AUDIO_TIME_DIM])
                mw[idx] = CW.IndexListContextWindow(
                    sub.index_list, dim=AUDIO_TIME_DIM, total_frames=total,
                    context_overlap=getattr(sub, "context_overlap", 0))
            return out

        state.prepare_window = prepare_window
        state._h3_windowing_patched = True

    # ---- 3. fuse each modality on ITS OWN axis ---------------------------- #
    # combine_context_window_results builds ONE weights tensor from the primary
    # dim and multiplies EVERY modality's output by it:
    #     weights_tensor = match_weights_to_dim(weights, x_in, self.dim, ...)
    #     for i in range(len(sub_conds_out)):
    #         window.add_window(conds_final[i], sub_conds_out[i] * weights_tensor)
    # For H3 that meets the audio's stereo axis and dies with
    #   "size of tensor a (2) must match the size of tensor b (150) at dim 2"
    # LTXAV survives it only because its audio time is also dim 2.
    handler = getattr(CW, "IndexListContextHandler", None)
    if handler is not None and not getattr(handler, "_h3_windowing_patched", False):
        original_combine = handler.combine_context_window_results

        def combine(self, x_in, sub_conds_out, sub_conds, window, window_idx,
                    total_windows, timestep, conds_final, counts_final,
                    biases_final):
            mw = getattr(window, "modality_windows", None) or {}
            # Only take over when a modality genuinely sits on another axis.
            # Everything else -- every other model -- keeps core's path exactly.
            off_axis = {i: w for i, w in mw.items()
                        if i < len(sub_conds_out)
                        and getattr(w, "dim", self.dim) != self.dim}
            if not off_axis or self.fuse_method.name == CW.ContextFuseMethods.RELATIVE:
                return original_combine(self, x_in, sub_conds_out, sub_conds,
                                        window, window_idx, total_windows,
                                        timestep, conds_final, counts_final,
                                        biases_final)

            weights = CW.get_context_weights(
                window.context_length, x_in.shape[self.dim], window.index_list,
                self, sigma=timestep, context_overlap=window.context_overlap)
            wt = CW.match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)

            for i in range(len(sub_conds_out)):
                sub = off_axis.get(i)
                if sub is None:
                    window.add_window(conds_final[i], sub_conds_out[i] * wt)
                    window.add_window(counts_final[i], wt)
                    continue
                d = sub.dim
                w_i = CW.get_context_weights(
                    len(sub.index_list), conds_final[i].shape[d], sub.index_list,
                    self, sigma=timestep,
                    context_overlap=getattr(sub, "context_overlap", 0))
                wt_i = CW.match_weights_to_dim(w_i, conds_final[i], d,
                                               device=x_in.device)
                sub.add_window(conds_final[i], sub_conds_out[i] * wt_i, dim=d)
                sub.add_window(counts_final[i], wt_i, dim=d)

            for cb in comfy.patcher_extension.get_all_callbacks(
                    CW.IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS,
                    self.callbacks):
                cb(self, x_in, sub_conds_out, sub_conds, window, window_idx,
                   total_windows, timestep, conds_final, counts_final,
                   biases_final)

        handler.combine_context_window_results = combine
        handler._h3_windowing_patched = True

    _PATCHED = True
    logging.info("[h3_toolkit] MiniMaxH3 context windowing patched: modality "
                 "mapping on the real (1,4,4,4,4) frame grid, audio sliced on "
                 "dim %d", AUDIO_TIME_DIM)
    return True


class H3EnableContextWindows:
    """Make ComfyUI's context windows work on MiniMax-H3.

    Wire the model through this once, anywhere before the sampler, then use core's
    `Context Windows (Manual)` as normal. It patches two hooks onto the H3 model
    class and corrects the window builder's assumption about where the audio
    latent's time axis lives. The MODEL passes through unchanged.

    What this buys: windowed sampling over ONE latent. No join, no trim, no
    per-chunk VAE round-trip -- and because the denoise mask is applied outside
    the windowing, a masked V2V swap over a long clip becomes a single sampler
    pass.

    What it does NOT buy: memory. Context windows bound the ATTENTION cost, but
    the full latent stays resident on device. A clip too large to hold is still
    too large to hold.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    EXPERIMENTAL = True
    DESCRIPTION = ("Patch MiniMax-H3 so ComfyUI's Context Windows can window it. "
                   "Pass the model through, then use Context Windows (Manual).")

    def go(self, model):
        ok = patch_h3_context_windows()
        if ok:
            text = ("H3 context windowing ENABLED.\n"
                    "  video windows on dim 2; audio remapped onto dim 3\n"
                    "  video->audio mapping uses the real (1,4,4,4,4) frame grid,\n"
                    "  not a proportional split — core's generic mapping is off by\n"
                    "  up to 3 pixel frames, worst at the clip start.\n"
                    "  Now wire Context Windows (Manual) into the sampler.\n"
                    "  NOTE this bounds attention cost, NOT memory: the whole\n"
                    "  latent stays resident.")
        else:
            text = ("Could not patch — comfy.context_windows or MiniMaxH3 was not "
                    "importable on this build. Context windowing is unavailable.")
        logging.info("H3EnableContextWindows: %s", text.splitlines()[0])
        return {"ui": {"h3char": [text]}, "result": (model, text)}


NODE_CLASS_MAPPINGS = {"H3EnableContextWindows": H3EnableContextWindows}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EnableContextWindows": "H3 Enable Context Windows"}
