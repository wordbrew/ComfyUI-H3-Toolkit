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

  That assumption is baked into six sites across two core classes. It was a core
  PATCH until 2026-08-30 -- `patches/h3-modality-dim-context-windows.patch`,
  which threaded a `modality_dims` list through both. A ComfyUI update reverts a
  core patch silently and the symptom is a bad render rather than an error, so it
  is a SUBCLASS now: `H3WindowingState` and `H3ContextHandler` below, installed
  into `model.model_options["context_handler"]` by `H3ContextWindows`. Core needs
  no edit at all.

  The handler is only an object in a dict, read back in samplers.py, so a
  subclass is all it takes. That route is MMH3Tools' -- `nodes_windows.py` in
  ComfyUI-MMH3Tools solves the same problem the same way, and this follows its
  structure.

  The cost is that four of the overrides are COPIES of core methods with the dim
  changed, so they drift when core changes. Each says which core method it
  mirrors and keeps the rest of the body line-for-line, so a future diff reads.
  test_windowing.py checks every override against core's own source.

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
    """Always True: this pack applies the offset, not core.

    It used to read core's source for a `window_start` parameter, from when the
    fix lived in a patch file. video.py's PackedLayout subclass shifts the
    finished position table itself now, so the capability travels with the pack
    and a ComfyUI update cannot take it away.
    """
    return True


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


# Built on first use by _context_window_classes(), not at import.
H3WindowingState = None
H3ContextHandler = None


def _context_window_classes():
    """H3's WindowingState and ContextHandler subclasses. -> (state, handler)

    Built inside a function because `class X(WindowingState)` needs
    comfy.context_windows AT IMPORT TIME, and importing that pulls in
    model_management and initialises CUDA. This module also holds the pure
    window arithmetic that test_windowing.py runs with no torch at all, and a
    module-level comfy import would take that away. The classes are assigned to
    the module globals on the first call, so `windowing.H3ContextHandler` is a
    real name from then on.
    """
    global H3WindowingState, H3ContextHandler
    if H3ContextHandler is not None:
        return H3WindowingState, H3ContextHandler

    import dataclasses

    import torch

    import comfy.patcher_extension
    import comfy.utils
    from comfy.context_windows import (ContextFuseMethods, IndexListCallbacks,
                                       IndexListContextHandler,
                                       IndexListContextWindow, WindowingState,
                                       apply_freenoise, get_context_weights,
                                       get_shape_for_dim, match_weights_to_dim)

    @dataclasses.dataclass
    class H3WindowingState(WindowingState):
        """Core's WindowingState with a temporal dim PER MODALITY.

        Core carries one `dim` and uses it for every modality. H3's audio puts
        time on dim 3, so every place core writes `self.dim` against a
        non-primary latent is a place it would take the stereo pair instead.
        """

        # per-modality temporal dim; defaults to `dim` for every modality.
        # A BARE `list`, not a string annotation: dataclasses resolves a string
        # one through sys.modules[cls.__module__] to look for ClassVar, which
        # raises when this module is loaded outside the package.
        modality_dims: list = None

        def __post_init__(self):
            if self.modality_dims is None:
                self.modality_dims = [self.dim] * len(self.latents)

        def prepare_window(self, window, model):
            """MIRRORS comfy.context_windows.WindowingState.prepare_window.

            A COPY of core's body, so it DRIFTS if core changes that method. The
            only edits are inside the modality loop: the per-modality frame count
            is read on that modality's own dim, and the window it builds is given
            that dim rather than the primary's. Everything else is core's, line
            for line, so a future diff against core stays readable.
            """
            if not self.is_multimodal:
                return window

            x = self.latents[0]
            primary_total = self.latent_shapes[0][self.dim]
            primary_overlap = window.context_overlap
            map_shapes = self.latent_shapes
            if x.size(self.dim) != primary_total:
                map_shapes = list(self.latent_shapes)
                video_shape = list(self.latent_shapes[0])
                video_shape[self.dim] = x.size(self.dim)
                map_shapes[0] = torch.Size(video_shape)
            try:
                per_modality_indices = model.map_context_window_to_modalities(
                    window.index_list, map_shapes, self.dim)
            except AttributeError:
                raise NotImplementedError(
                    f"{type(model).__name__} must implement "
                    f"map_context_window_to_modalities for multimodal context "
                    f"windows.")
            modality_windows = {}
            for mod_idx in range(1, len(self.latents)):
                mod_dim = self.modality_dims[mod_idx]
                modality_total_frames = self.latents[mod_idx].shape[mod_dim]
                ratio = (modality_total_frames / primary_total
                         if primary_total > 0 else 1)
                modality_overlap = max(round(primary_overlap * ratio), 0)
                modality_windows[mod_idx] = IndexListContextWindow(
                    per_modality_indices[mod_idx], dim=mod_dim,
                    total_frames=modality_total_frames,
                    context_overlap=modality_overlap)
            return IndexListContextWindow(
                window.index_list, dim=self.dim, total_frames=x.shape[self.dim],
                modality_windows=modality_windows,
                context_overlap=primary_overlap)

        def strip_guide_frames(self, out_per_modality, guide_frame_counts,
                               window):
            """MIRRORS comfy.context_windows.WindowingState.strip_guide_frames.

            A COPY of core's body; it DRIFTS if core changes that method. One
            edit: the narrow runs on the modality's own dim. H3 has no guide
            frames today -- they come from LTXV's `guide_attention_entries` --
            so this never fires here, but leaving core's version in place would
            trim audio along its stereo pair the day it does.
            """
            for idx in range(len(self.latents)):
                if guide_frame_counts[idx] > 0:
                    window_len = len(window.get_window_for_modality(idx).index_list)
                    for ci in range(len(out_per_modality)):
                        out_per_modality[ci][idx] = out_per_modality[ci][idx].narrow(
                            self.modality_dims[idx], 0, window_len)

        def _inject_guide_frames(self, latent_slice, window, modality_idx=0):
            """WRAPS comfy.context_windows.WindowingState._inject_guide_frames.

            Core uses `self.dim` in that method for exactly one thing: the axis
            the guide frames are concatenated onto, which has to be the
            modality's own. Rather than copy 30 lines to change two, this swaps
            `self.dim` for the duration of the call -- a no-op for the primary
            modality, since its dim IS the handler's. Not a copy, so it does not
            drift; if core ever uses `self.dim` in there to mean the PRIMARY
            axis, this becomes wrong, and that is the one thing to check.
            """
            saved = self.dim
            self.dim = self.modality_dims[modality_idx]
            try:
                return super()._inject_guide_frames(latent_slice, window,
                                                    modality_idx)
            finally:
                self.dim = saved

    class H3ContextHandler(IndexListContextHandler):
        """Core's IndexListContextHandler, windowing each modality on its axis.

        Everything not listed here is core's. What changes: the windowing state
        it builds carries per-modality dims; the fuse accumulators are sized on
        each modality's own axis; the fuse itself uses the WINDOW's dim, since
        it is called once per modality with that modality's window; and
        FreeNoise shuffles each modality on its own axis.
        """

        def _get_modality_dims(self, model, latent_shapes, count):
            """Temporal dim of each modality's latent. NEW -- no core original.

            The primary modality always uses the handler's configured dim; a
            non-primary one may report a different one via the model. The hook
            is optional: `patch_h3_context_windows()` installs H3's on the
            MiniMaxH3 class, and any model without it keeps core's behaviour of
            one dim for everything.
            """
            dims = [self.dim] * count
            modality_dim = getattr(model, "context_modality_dim", None)
            if modality_dim is not None:
                for i in range(1, count):
                    dims[i] = modality_dim(i, latent_shapes, self.dim)
            return dims

        def _apply_freenoise(self, noise, conds, seed, model=None):
            """MIRRORS comfy.context_windows.IndexListContextHandler._apply_freenoise.

            A COPY of core's body; it DRIFTS if core changes that method. The
            edits are in the multimodal branch: each modality's total and its
            shuffle both run on that modality's dim. On the stereo axis the
            ratio comes out 2/T, which gives a context length of 1 and permutes
            the left channel into the right.

            `model` is appended rather than inserted: core's own sampler wrapper
            calls this with three arguments, and it has to keep working if
            anything but our wrapper reaches it. Without a model the dims fall
            back to the handler's, which is core's behaviour.
            """
            guide_entries = self._get_guide_entries(conds)
            guide_count = (sum(e["latent_shape"][0] for e in guide_entries)
                           if guide_entries else 0)

            latent_shapes = self._get_latent_shapes(conds)
            if latent_shapes is not None and len(latent_shapes) > 1:
                modalities = comfy.utils.unpack_latents(noise, latent_shapes)
                modality_dims = self._get_modality_dims(model, latent_shapes,
                                                        len(modalities))
                primary_total = latent_shapes[0][self.dim]
                primary_video_count = modalities[0].size(self.dim) - guide_count
                apply_freenoise(modalities[0].narrow(self.dim, 0, primary_video_count),
                                self.dim, self.context_length,
                                self.context_overlap, seed)
                for i in range(1, len(modalities)):
                    mod_dim = modality_dims[i]
                    mod_total = latent_shapes[i][mod_dim]
                    ratio = mod_total / primary_total if primary_total > 0 else 1
                    mod_ctx_len = max(round(self.context_length * ratio), 1)
                    mod_ctx_overlap = max(round(self.context_overlap * ratio), 0)
                    modalities[i] = apply_freenoise(modalities[i], mod_dim,
                                                    mod_ctx_len, mod_ctx_overlap,
                                                    seed)
                noise, _ = comfy.utils.pack_latents(modalities)
                return noise
            video_count = noise.size(self.dim) - guide_count
            apply_freenoise(noise.narrow(self.dim, 0, video_count), self.dim,
                            self.context_length, self.context_overlap, seed)
            return noise

        def _build_window_state(self, x_in, conds, model):
            """WRAPS comfy.context_windows.IndexListContextHandler._build_window_state.

            Core's state, re-made as H3's with the per-modality dims attached.
            Copies field by field off the dataclass rather than by name, so a
            new field in core's WindowingState travels here without an edit.
            """
            st = super()._build_window_state(x_in, conds, model)
            fields = {f.name: getattr(st, f.name)
                      for f in dataclasses.fields(st)}
            fields["modality_dims"] = self._get_modality_dims(
                model, st.latent_shapes, len(st.latents))
            return H3WindowingState(**fields)

        def execute(self, calc_cond_batch, model, conds, x_in, timestep,
                    model_options):
            """MIRRORS comfy.context_windows.IndexListContextHandler.execute.

            A COPY of core's body; it DRIFTS if core changes that method. Copied
            rather than wrapped because the accumulator allocation is inline and
            it is what has to change: core sizes `counts` and `biases` on
            `self.dim`, which for H3's audio is the stereo pair -- extent 2
            against T40 -- and dies in the fuse with "size of tensor a (2) must
            match the size of tensor b (93)". The two edits are marked CHANGED;
            everything else is core's, line for line.
            """
            self._model = model
            self.set_step(timestep, model_options)

            window_state = self._build_window_state(x_in, conds, model)
            num_modalities = len(window_state.latents)

            context_windows = self.get_context_windows(
                model, window_state.latents[0], model_options)
            enumerated_context_windows = list(enumerate(context_windows))
            total_windows = len(enumerated_context_windows)

            # Initialize per-modality accumulators (length 1 for single-modality)
            # CHANGED: counts and biases are sized on each modality's own dim
            modality_dims = window_state.modality_dims
            accum = [[torch.zeros_like(m) for _ in conds]
                     for m in window_state.latents]
            if self.fuse_method.name == ContextFuseMethods.RELATIVE:
                counts = [[torch.ones(get_shape_for_dim(m, modality_dims[mi]),
                                      device=m.device) for _ in conds]
                          for mi, m in enumerate(window_state.latents)]
            else:
                counts = [[torch.zeros(get_shape_for_dim(m, modality_dims[mi]),
                                       device=m.device) for _ in conds]
                          for mi, m in enumerate(window_state.latents)]
            biases = [[([0.0] * m.shape[modality_dims[mi]]) for _ in conds]
                      for mi, m in enumerate(window_state.latents)]

            for callback in comfy.patcher_extension.get_all_callbacks(
                    IndexListCallbacks.EXECUTE_START, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

            # accumulate results from each context window
            for enum_window in enumerated_context_windows:
                results = self.evaluate_context_windows(
                    calc_cond_batch, model, x_in, conds, timestep, [enum_window],
                    model_options, window_state=window_state,
                    total_windows=total_windows)
                for result in results:
                    # result.sub_conds_out is per-cond, per-modality
                    for mod_idx in range(num_modalities):
                        mod_out = [result.sub_conds_out[ci][mod_idx]
                                   for ci in range(len(conds))]
                        modality_window = result.window.get_window_for_modality(mod_idx)
                        self.combine_context_window_results(
                            window_state.latents[mod_idx], mod_out,
                            result.sub_conds, modality_window,
                            result.window_idx, total_windows, timestep,
                            accum[mod_idx], counts[mod_idx], biases[mod_idx])

            # fuse accumulated results into final conds
            try:
                result_out = []
                for ci in range(len(conds)):
                    finalized = []
                    for mod_idx in range(num_modalities):
                        if self.fuse_method.name != ContextFuseMethods.RELATIVE:
                            accum[mod_idx][ci] /= counts[mod_idx][ci]
                        f = accum[mod_idx][ci]

                        # if guide frames were injected, append them to the end
                        # of the fused latents for the next step
                        if window_state.guide_latents[mod_idx] is not None:
                            # CHANGED: concat on the modality's own dim
                            f = torch.cat([f, window_state.guide_latents[mod_idx]],
                                          dim=modality_dims[mod_idx])
                        finalized.append(f)

                    # pack modalities together if needed
                    if window_state.is_multimodal and len(finalized) > 1:
                        packed, _ = comfy.utils.pack_latents(finalized)
                    else:
                        packed = finalized[0]

                    result_out.append(packed)
                return result_out
            finally:
                for callback in comfy.patcher_extension.get_all_callbacks(
                        IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                    callback(self, model, x_in, conds, timestep, model_options)

        def combine_context_window_results(self, x_in, sub_conds_out, sub_conds,
                                           window, window_idx, total_windows,
                                           timestep, conds_final, counts_final,
                                           biases_final):
            """MIRRORS IndexListContextHandler.combine_context_window_results.

            A COPY of core's body; it DRIFTS if core changes that method. One
            edit, at the top: this is called once PER MODALITY with that
            modality's own window and latent, so the axis to work on is the
            window's dim, not the handler's. Core builds the fuse weights on
            `self.dim`, which sizes a 93-long weight vector onto the audio
            latent's stereo axis and raises "size of tensor a (2) must match the
            size of tensor b (93)". No new parameter is needed -- the window
            already knows.
            """
            dim = window.dim
            if self.fuse_method.name == ContextFuseMethods.RELATIVE:
                for pos, idx in enumerate(window.index_list):
                    # bias is the influence of a specific index in relation to
                    # the whole context window
                    bias = 1 - abs(idx - (window.index_list[0] + window.index_list[-1]) / 2) / ((window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                    bias = max(1e-2, bias)
                    # take weighted average relative to total bias of current idx
                    for i in range(len(sub_conds_out)):
                        bias_total = biases_final[i][idx]
                        prev_weight = (bias_total / (bias_total + bias))
                        new_weight = (bias / (bias_total + bias))
                        # account for dims of tensors
                        idx_window = tuple([slice(None)] * dim + [idx])
                        pos_window = tuple([slice(None)] * dim + [pos])
                        # apply new values
                        conds_final[i][idx_window] = conds_final[i][idx_window] * prev_weight + sub_conds_out[i][pos_window] * new_weight
                        biases_final[i][idx] = bias_total + bias
            else:
                # add conds and counts based on weights of fuse method
                weights = get_context_weights(window.context_length,
                                              x_in.shape[dim],
                                              window.index_list, self,
                                              sigma=timestep,
                                              context_overlap=window.context_overlap)
                weights_tensor = match_weights_to_dim(weights, x_in, dim,
                                                      device=x_in.device)
                for i in range(len(sub_conds_out)):
                    window.add_window(conds_final[i],
                                      sub_conds_out[i] * weights_tensor)
                    window.add_window(counts_final[i], weights_tensor)

            for callback in comfy.patcher_extension.get_all_callbacks(
                    IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS,
                    self.callbacks):
                callback(self, x_in, sub_conds_out, sub_conds, window,
                         window_idx, total_windows, timestep, conds_final,
                         counts_final, biases_final)

    return H3WindowingState, H3ContextHandler


def _h3_sampler_sample_wrapper(executor, guider, sigmas, extra_args, callback,
                               noise, *args, **kwargs):
    """MIRRORS comfy.context_windows._sampler_sample_wrapper.

    A COPY of core's, with the model handed to `_apply_freenoise`. Core's
    wrapper calls it with three arguments, and the model is what says which
    axis each modality's time is on -- without it FreeNoise would shuffle H3's
    audio along the stereo pair. The wrapper is the only place a model is in
    scope this early: FreeNoise runs on the noise before sampling starts, so the
    handler has not been handed one yet.
    """
    model_options = extra_args.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in sampler_sample_wrapper; "
                        "this should never happen, something went wrong.")
    handler = model_options.get("context_handler", None)
    if handler is None:
        raise Exception("context_handler not found in sampler_sample_wrapper; "
                        "this should never happen, something went wrong.")
    if not handler.freenoise:
        return executor(guider, sigmas, extra_args, callback, noise, *args,
                        **kwargs)

    conds = [guider.conds.get('positive', guider.conds.get('negative', []))]
    noise = handler._apply_freenoise(noise, conds, extra_args["seed"],
                                     guider.model_patcher.model)

    return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)


def create_h3_sampler_sample_wrapper(model):
    """Install `_h3_sampler_sample_wrapper` in place of core's.

    Its own key, so it cannot collide with core's if both ever end up on one
    ModelPatcher.
    """
    import comfy.patcher_extension
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        "H3ContextWindows_sampler_sample",
        _h3_sampler_sample_wrapper)


def patch_h3_context_windows():
    """Install H3's side of the modality-dim contract. -> (usable, message).

    ALSO installs video.py's PackedLayout subclass, which is what applies the
    per-window target offset. That patch used to be installed only by this
    pack's own conditioning nodes -- so a graph built on CORE's
    MiniMaxH3ReferenceToVideo never got it, every window stayed at the clip
    origin, and the only symptom was the flicker the offset exists to remove.
    Windowing depends on the subclass, so enabling windowing installs it.
    """
    try:
        import torch
        import comfy.model_base as MB
    except Exception as exc:
        return False, f"comfy.model_base not importable: {type(exc).__name__}"

    H3 = getattr(MB, "MiniMaxH3", None)
    if H3 is None:
        return False, "MiniMaxH3 not found in comfy.model_base."

    from .video import patch_packed_layout
    if not patch_packed_layout():
        return False, ("the H3 layout patch could not be installed, so the "
                       "per-window target offset has nothing to apply it.")

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
                    # Hand off to the layout rather than to core: PackedLayout is
                    # built inside extra_conds, which passes it no window
                    # position, and patching that call site is exactly what a
                    # ComfyUI update reverts. video.py takes this on the next
                    # build and clears it.
                    from .video import set_window_start
                    set_window_start(start)
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

    return True, (
        "H3 context windowing ENABLED.\n"
        "  H3 hooks installed on the model:\n"
        "    audio time resolved to dim 3, not the stereo pair at dim 2\n"
        "    video->audio mapped on the real (1,4,4,4,4) frame grid, not\n"
        "    proportionally — proportional is off by up to 3 pixel frames,\n"
        "    worst at the clip start\n"
        "  Now wire H3 Context Windows into the sampler.\n"
        "  NOT core's Context Windows (Manual): the per-modality axis lives in\n"
        "  this pack's handler subclass, and core's own handler would slice H3's\n"
        "  audio on its stereo pair. No ComfyUI version check applies — nothing\n"
        "  in core has to change.\n"
        "\n"
        "  Bounds ATTENTION cost, not memory — the whole latent stays\n"
        "  resident.\n"
        "  REFERENCE blocks survive being windowed: proven 2026-08-29, they\n"
        "  travel with each per-window conditioning and hold identity.\n"
        "  KEYFRAMES are handled — dropped outside the window, and with\n"
        "  absolute positions their indices are kept rather than rebased —\n"
        "  but have never been run in a windowed render. Check the seams.")


class H3EnableContextWindows:
    """Make ComfyUI's context windows work on MiniMax-H3.

    Pass the model through once before the sampler, then use `H3 Context
    Windows`. The MODEL passes through unchanged; this only installs H3's side
    of the modality-dim contract on the MiniMaxH3 class.

    `H3 Context Windows` calls this itself, so wiring it is belt and braces --
    what it buys is the `info` text, which says what got installed rather than
    letting a render find out.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    EXPERIMENTAL = True
    DESCRIPTION = ("Install MiniMax-H3's context-window hooks. Reports what went "
                   "in; H3 Context Windows installs them too.")

    def go(self, model):
        usable, text = patch_h3_context_windows()
        logging.info("H3EnableContextWindows: %s", text.splitlines()[0])
        return {"ui": {"h3char": [text]}, "result": (model, text)}


class H3ContextWindows:
    """Context windows for H3, in FRAMES, phase-aligned, on the right axis.

    Core's `Context Windows (Manual)` has four ways to get H3 silently wrong,
    and this node closes all four:

      ITS HANDLER WINDOWS EVERY MODALITY ON ONE DIM. H3's audio latent is
      [B, 32, 2, T] -- dim 2 is the stereo pair. This node installs
      H3ContextHandler instead, which gives each modality its own axis.

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
            "model": ("MODEL", {"tooltip": "H3's model hooks go on here "
                                           "automatically. H3 Enable Context "
                                           "Windows reports what went in."}),
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
            # 167 layout builds, one cond_t. The offset is applied by video.py's
            # PackedLayout subclass, so it needs no core patch.
            "absolute_window_positions": ("BOOLEAN", {"default": False,
                                          "tooltip": "Give each window its real "
                                                     "position on the clip's "
                                                     "timeline instead of the "
                                                     "clip origin, so a later "
                                                     "window reads as 'this clip, "
                                                     "N frames in' rather than "
                                                     "'another clip'."}),
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
            notes.append("absolute_window_positions asked for, but the "
                         "per-window offset is not available — video.py's "
                         "PackedLayout subclass did not install. Running with "
                         "origin-positioned windows.")
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

        # The MODEL hooks -- the per-modality dim, the frame-grid mapping, the
        # payload rebasing -- live on the MiniMaxH3 class, and the handler is
        # useless without them: with no `context_modality_dim` it falls back to
        # one dim for every modality, which is core's behaviour and H3's bug.
        # Installing them here as well as in H3EnableContextWindows costs an
        # idempotent call and removes a graph where the enable node is simply
        # missing and the render is silently wrong.
        hooks_ok, hooks_msg = patch_h3_context_windows()
        if not hooks_ok:
            notes.append(hooks_msg)

        # Install OUR handler, not core's `Context Windows (Manual)`.
        #
        # This used to resolve that node and call it, which built core's
        # IndexListContextHandler -- correct only while a core PATCH was applied
        # to teach it H3's per-modality axes. A ComfyUI update reverted that
        # patch silently and the symptom was a bad render, not an error.
        # H3ContextHandler carries the same behaviour as a subclass, so nothing
        # in core has to change. Everything else here is what that node did:
        # clone, set model_options["context_handler"], add the sampling wrapper.
        try:
            from comfy.context_windows import (create_prepare_sampling_wrapper,
                                               get_matching_context_schedule,
                                               get_matching_fuse_method)
            _, handler_cls = _context_window_classes()
        except Exception as exc:
            msg = (f"comfy.context_windows is not importable "
                   f"({type(exc).__name__}: {exc}) — windowing is unavailable "
                   f"here.")
            return {"ui": {"h3char": [msg]}, "result": (model, msg)}

        patched = model.clone()
        patched.model_options["context_handler"] = handler_cls(
            context_schedule=get_matching_context_schedule(schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=w_lat,
            context_overlap=o_lat,
            context_stride=1,
            closed_loop=False,
            dim=VIDEO_TIME_DIM,
            freenoise=freenoise,
            cond_retain_index_list=[],
            split_conds_to_windows=bool(split_conds_to_windows),
            latent_retain_index_list=[],
            causal_window_fix=bool(causal_window_fix))
        # makes the VRAM estimate budget one window rather than the whole clip
        create_prepare_sampling_wrapper(patched)
        if freenoise:
            # ours rather than core's: it hands _apply_freenoise the model, which
            # is what says audio's time is on dim 3
            create_h3_sampler_sample_wrapper(patched)

        base = getattr(patched, "model", None)
        if base is not None:
            setattr(base, ABSOLUTE_FLAG, absolute)

        # One line per RUN saying what was actually installed. The afternoon this
        # cost was spent on a widget that read standard_static while the handler
        # ran uniform, and on a mode that turned itself off when the node was
        # cached -- both invisible from the outside, both one line to catch.
        logging.info("H3 context windows: %s frames (%s latent), overlap %s (%s), "
                     "stride %s | schedule %s | fuse %s | freenoise %s | "
                     "causal_fix %s | absolute positions %s | split conds %s",
                     wf, w_lat, of, o_lat, wf - of, schedule, fuse_method,
                     freenoise, causal_window_fix, absolute,
                     split_conds_to_windows)

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
