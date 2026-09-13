"""Replacing a masked region of an existing video, and holding the rest still.

Everything here pins known content and lets the sampler work the remainder. The
pinning is in LATENT space, which is what makes the geometry fiddly: a latent cell
covers 16x16 pixels and a latent frame covers several pixel frames, so a mask drawn
per pixel per frame has to be reduced to that grid before it means anything.

  - A SPATIAL mask edge is a normal inpainting problem and the model blends across
    it. A TEMPORAL one is not: every join cut traced over nine rounds sat exactly at
    the frame where pinned content stopped, and feathering never moved it. That is
    why H3LatentPin carries a warning and H3MaskInpaint does not.
"""

import logging

import torch
import torch.nn.functional as Fn

from .avlatent import av
from .chunkplan import snap_context
from .geometry import cover_crop, crop_to_multiple
from .timing import (FPS, FRAME_PER_TOKEN, align_frames, audio_t,
                     av_aligned_runs_through,
                     describe, frame_groups, is_av_aligned, snap_av_aligned)

CATEGORY = "MiniMax H3/mask"


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
                               "tooltip": "Grow the mask in LATENT cells, after the "
                                          "reduction. Each cell is 16 px, so 2 is ~32 px "
                                          "of margin — coarse. `grow_px` is the finer "
                                          "control."}),
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
            "forget_mask": ("MASK", {
                "tooltip": "Greyscale. How much each area FORGETS the source it started "
                           "from. Black = remember it fully (what happens with nothing "
                           "connected); white = start from noise with no memory. Only "
                           "meaningful inside the main mask.\n\n"
                           "This is the knob for 'the model keeps the original hair "
                           "colour even though it is masked'. Below denoise 1.0 the free "
                           "region still STARTS from the source, and hue survives "
                           "denoising better than anything else, so a fraction of a "
                           "percent is enough to carry blonde through. Painting the hair "
                           "white here removes that memory without touching denoise, so "
                           "the body keeps the source residual that holds its pose."}),
            "forget_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                "step": 0.05,
                                "tooltip": "Global multiplier on forget_mask, so you can "
                                           "sweep it without re-authoring the mask. 0 "
                                           "reproduces the old behaviour exactly."}),
            "grow_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                        "tooltip": "Grow the mask in PIXELS, before it is reduced to "
                                   "latent space. Finer than `dilate`, which can only "
                                   "move in 16 px jumps because it works on latent "
                                   "cells. Prefer this one and leave dilate at 0."}),
            "token_snap": ("BOOLEAN", {"default": False,
                           "tooltip": "Snap the mask to the model's 2x2 patch grid — 32 "
                                      "px instead of 16. TESTED AND NOT OBSERVABLE: at "
                                      "denoise 0.45, 0.70 and 1.0 it made no visible "
                                      "difference, and it costs ~2.4% more regenerated "
                                      "area, so it is off.\n\n"
                                      "The theory was that the sampler pins per latent "
                                      "CELL while the DiT reasons per 2x2 PATCH, so a "
                                      "finer mask splits tokens. It does — 4.7% of them "
                                      "on a subject-shaped mask. But the pinning happens "
                                      "BEFORE the model call and fills the pinned half "
                                      "with correctly-noised source, so both halves are "
                                      "valid latents and the model just sees an edge "
                                      "inside a token, which is ordinary. Kept as an "
                                      "option in case it matters somewhere it was not "
                                      "tested."}),
            "sigmas": ("SIGMAS", {
                "tooltip": "Optional, from the same scheduler feeding the sampler. Lets "
                           "the fill be variance-corrected for the sigma sampling will "
                           "actually start at. Without it the forgotten region comes out "
                           "slightly UNDER-noised, which the model reads as further along "
                           "than it is and answers by over-sharpening."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Regenerate a masked region of an existing video while pinning "
                   "everything outside it. Feed a SAM mask and reference anchors to "
                   "replace a person without touching the scene. forget_mask controls "
                   "how much of the region remembers the source it started from.")

    def go(self, latent, vae, source_images, mask, dilate, feather, invert, keep_audio,
           grow_px=0, token_snap=True, audio_vae=None, source_audio=None,
           forget_mask=None, forget_strength=1.0, sigmas=None):
        import comfy.nested_tensor
        import torch.nn.functional as Fn

        video, aud = av(latent["samples"])
        lt, lh, lw = video.shape[2], video.shape[3], video.shape[4]

        # CONFORM RATHER THAN REFUSE, BUT NEVER STRETCH. The latent's shape is set by
        # the H3 node's width/height/length; the source is whatever clip you loaded,
        # and the two numbers live on different nodes with nothing keeping them in
        # step — so refusing just moves the problem. But how we conform matters:
        #
        #   crop   changes framing, leaves every retained pixel untouched
        #   scale  keeps framing, resamples every pixel
        #   stretch  distorts anatomy AND resamples          <- never
        #
        # In an inpaint most of the output IS the source: everything outside the mask
        # is pinned from these exact pixels. So we centre-crop to the target ASPECT
        # first (free — no resampling), and only then rescale if the size still
        # differs. An earlier version here stretched to fit, which is the worst of the
        # three: it softens the pixels you are keeping and hands the model a squashed
        # body to match, fighting everything it knows about anatomy.
        #
        # Frame COUNT is not resamplable (17n+5 grid, several pixel frames per latent
        # frame), so that still errors.
        sw, sh = source_images.shape[2], source_images.shape[1]
        tw, th = lw * 16, lh * 16
        src = source_images
        m3 = mask if mask.dim() == 3 else mask.unsqueeze(0)
        fg3 = None
        if forget_mask is not None and float(forget_strength) > 0:
            fg3 = forget_mask if forget_mask.dim() == 3 else forget_mask.unsqueeze(0)
        if (sw, sh) != (tw, th):
            x0, y0, cw, chh = cover_crop(sw, sh, tw, th)
            if (cw, chh) != (sw, sh):
                src = src[:, y0:y0 + chh, x0:x0 + cw, :]
                m3 = m3[..., y0:y0 + chh, x0:x0 + cw]
                if fg3 is not None:
                    fg3 = fg3[..., y0:y0 + chh, x0:x0 + cw]
                logging.warning(
                    "H3MaskInpaint: source %dx%d (%.2f:1) does not match the latent's "
                    "aspect %dx%d (%.2f:1) — centre-cropped to %dx%d rather than "
                    "stretching. Content outside that crop is GONE; set the H3 node's "
                    "width/height to the source's aspect to keep the full frame.",
                    sw, sh, sw / max(1, sh), tw, th, tw / max(1, th), cw, chh)
            if (cw, chh) != (tw, th):
                logging.info("H3MaskInpaint: rescaling source %dx%d -> %dx%d (aspect "
                             "preserved)", cw, chh, tw, th)
                src = Fn.interpolate(src.movedim(-1, 1), size=(th, tw), mode="bicubic",
                                     align_corners=False,
                                     antialias=True).clamp(0, 1).movedim(1, -1)
                m3 = Fn.interpolate(m3.unsqueeze(1), size=(th, tw),
                                    mode="nearest").squeeze(1)
                if fg3 is not None:
                    # bilinear, not nearest: this one carries meaningful mid-tones
                    fg3 = Fn.interpolate(fg3.unsqueeze(1), size=(th, tw),
                                         mode="bilinear", align_corners=False).squeeze(1)
        mask = m3

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

        # Growing HERE, in pixels, moves the edge one pixel at a time. Growing after
        # the reduction can only move it in 16px jumps, which on a hand or a strand
        # of hair is the difference between a margin and a blob. Separable so a
        # large radius stays cheap.
        if grow_px > 0:
            r = int(grow_px)
            m = Fn.max_pool3d(m, (1, 2 * r + 1, 1), stride=1, padding=(0, r, 0))
            m = Fn.max_pool3d(m, (1, 1, 2 * r + 1), stride=1, padding=(0, 0, r))

        # UNION, not resample: any pixel frame contributing to a latent frame counts.
        #
        # Spatially that is an exact 16x16 block — the source was conformed to
        # lh*16 x lw*16 above, so the blocks divide evenly.
        m = Fn.max_pool3d(m, kernel_size=(1, 16, 16), stride=(1, 16, 16))
        #
        # TEMPORALLY IT IS NOT EVEN, and this used to be wrong. `adaptive_max_pool3d`
        # splits T pixel frames into lt EQUAL buckets, but the VAE's grouping is
        # FRAME_PER_TOKEN = (1,4,4,4,4): every fifth latent frame covers ONE pixel
        # frame and the rest cover four. Equal buckets put 4-5 frames of unioned mask
        # onto the single-frame tokens and shift every boundary by up to two frames
        # (83 ms at 24 fps), cyclically. Fast-moving occluders smeared across their
        # own path on exactly the tokens that should have been sharpest.
        sizes = frame_groups(lt)
        if sum(sizes) != m.shape[2]:
            raise ValueError(
                f"mask has {m.shape[2]} frame(s) but {lt} latent frames need "
                f"{sum(sizes)}. The mask must be the same length as the source clip "
                f"— put H3 Match Source Clip in front of both.")
        m = torch.stack([g.amax(dim=2) for g in torch.split(m, sizes, dim=2)], dim=2)

        if dilate > 0:
            k = dilate * 2 + 1
            m = Fn.max_pool3d(m, kernel_size=(1, k, k), stride=1,
                              padding=(0, dilate, dilate))
        if feather > 0:
            r = max(1, int(round(feather * 3)))
            k = r * 2 + 1
            # count_include_pad=False, or the zero padding outside the image is
            # averaged in and drags the mask down at the FRAME edges: a solid 1.0
            # mask came out at 0.444 in the corners, so the outer ring of cells was
            # partially pinned and blended generated content with source in a band
            # around the whole picture.
            m = Fn.avg_pool3d(m, kernel_size=(1, k, k), stride=1, padding=(0, r, r),
                              count_include_pad=False)
            m = m.clamp(0, 1)

        if token_snap:
            # The sampler pins per latent CELL; the DiT reasons per 2x2 PATCH. A mask
            # finer than the patch gives it a token that is half pinned and half free,
            # and it has no sub-token resolution to resolve that with. Max, so a patch
            # any part of which should regenerate, regenerates whole — the same rule
            # the engine's own _mask_row_values applies.
            ph, pw = lh % 2, lw % 2
            q = m if (ph == 0 and pw == 0) else Fn.pad(m, (0, pw, 0, ph),
                                                       mode="replicate")
            q = Fn.max_pool3d(q, (1, 2, 2), stride=(1, 2, 2))
            q = q.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            m = q[..., :lh, :lw]

        mask_v = m.expand(video.shape[0], video.shape[1], lt, lh, lw).contiguous()
        mask_v = mask_v.to(video.device, video.dtype)
        known_v = z.to(video.device, video.dtype)

        # ---- forget: drop the source memory where asked -----------------------
        #
        # The sampler builds its starting point as
        #
        #     x_init = sigma * noise + (1 - sigma) * known_v
        #
        # over the WHOLE tensor, with no mask involved (KSAMPLER.sample calls
        # model_sampling.noise_scaling before any masking happens). So the free
        # region does not start blank — it starts holding (1 - sigma) of the source.
        # At denoise 0.45 under shift 12 that is 9% of it, and hue survives
        # denoising better than structure does, which is why a masked region can
        # still come back with the original hair colour.
        #
        # Replacing known_v with noise where `forget` is set removes that memory for
        # those cells only, so the rest of the mask keeps the residual that holds
        # pose and framing.
        forget_report = ""
        if fg3 is not None:
            f = fg3.float().unsqueeze(0).unsqueeze(0)          # [1,1,T,H,W]
            f = Fn.avg_pool3d(f, kernel_size=(1, 16, 16), stride=(1, 16, 16))
            if sum(sizes) != f.shape[2]:
                raise ValueError(
                    f"forget_mask has {f.shape[2]} frame(s) but the clip needs "
                    f"{sum(sizes)}. It must be the same length as source_images.")
            # MEAN over each group, not max: this mask's mid-tones are the point,
            # and a max would turn any touched cell fully white.
            f = torch.stack([g.mean(dim=2) for g in torch.split(f, sizes, dim=2)], dim=2)
            f = (f * float(forget_strength)).clamp(0, 1)
            f = f.expand_as(mask_v).to(video.device, video.dtype)
            # GATE on the mask, do not scale by it. known_v is not only the
            # initialisation, it is also the restore target:
            #
            #     x = x*mask + scale_latent_inpaint(..., latent_image)*(1 - mask)
            #
            # Where mask is 1 that term vanishes and the fill is never seen again.
            # But the mask is FEATHERED, so at the boundary mask sits between 0 and
            # 1, and there a cell holding 4.55x noise gets re-injected as if it were
            # real content on every step. Multiplying f by mask_v left exactly those
            # partial cells half-filled, which showed up as isolated 16px squares
            # scattered along the mask edge.
            #
            # So forget only where the cell is FULLY free. Edge cells keep the true
            # source, which is what a blend boundary wants anyway.
            f = f * (mask_v >= 0.999).to(f.dtype)

            # variance correction. x_init = sigma*n1 + (1-sigma)*k*n2 with n1, n2
            # independent, so Var = sigma^2 + (1-sigma)^2 * k^2. Setting that to 1
            # gives k = sqrt((1+sigma)/(1-sigma)). Without it the region is
            # UNDER-noised, which reads as further along the trajectory than it is
            # and comes back over-sharpened — the same failure as blending a feather
            # against nothing.
            s0 = None
            if sigmas is not None and len(sigmas) > 0:
                s0 = float(sigmas[0])
            if s0 is None:
                k = 1.0
                note = ("sigmas not connected, fill left at unit variance — the "
                        "forgotten region will be slightly under-noised")
            elif s0 >= 0.999:
                k = 1.0
                note = f"sigma {s0:.3f}: the source contributes nothing anyway"
            else:
                k = float((1.0 + s0) / (1.0 - s0)) ** 0.5
                note = f"sigma {s0:.3f}, fill scaled {k:.2f}x for unit variance"

            gen = torch.Generator(device="cpu").manual_seed(0x4F3D)
            fill = torch.randn(known_v.shape, generator=gen, dtype=torch.float32)
            fill = (fill * k).to(known_v.device, known_v.dtype)
            known_v = known_v * (1.0 - f) + fill * f

            covered = float(f.mean())
            forget_report = (f"forget: {covered * 100:.1f}% of the latent forgotten "
                             f"(strength {forget_strength:.2f}); {note}")
            logging.info("H3MaskInpaint: %s", forget_report)

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

        info = (f"{lw * 16}x{lh * 16}, {lt} latent frames; mask covers "
                f"{float(mask_v.mean()) * 100:.1f}% of the latent")
        if forget_report:
            info += "\n" + forget_report
        return {"ui": {"h3char": [info]}, "result": (out, info)}


class H3LatentPin:
    """Continue a clip: copy the previous clip's tail onto this one's opening.

    WHAT IT DOES TO THE LATENT
      H3's latent is a nested pair -- video [B,24,T,h,w] and audio [B,32,2,T40].
      The previous clip's LAST `overlap_frames` are written over this clip's
      FIRST, in both streams, and the noise mask is set to 0 there. The mask is
      re-applied by KSamplerX0Inpaint at EVERY denoise step, so those positions
      are stamped back 10 or 20 times and cannot drift: the model sees fixed
      content it has to continue from, and only the frames after the pin are
      generated. Nothing is added to the packed sequence -- no extra rows, no
      token cost. It is the target tensor with a pre-filled, pinned head.

    WHY THE OVERLAP MUST BE A LEGAL RUN
      This node used to carry a warning that it produced a visible cut where the
      pin ended, and the cause was here: `overlap_frames` was a free integer.
      The VAE groups pixel frames (1,4,4,4,4), so what a latent step covers
      depends on its index mod 5. Legal runs (17n+5) are 2 mod 5 latent steps,
      which means the last k steps of one clip and the first k of the next start
      at the SAME phase and a direct copy lines up. Ask for 30 and the old code
      took 7 steps -- 22 frames of coverage -- while slicing as though it were
      30, so the tail landed at a phase the prefix does not have. That is the
      cut. The widget is the grid now and the value is snapped either way.

    STRENGTH IS NOT A DIAL FOR SEAMS
      1.0 holds the prefix exactly, which is what a continuation wants. Lower
      values half-preserve it, and nine rounds of measurement found no setting
      between 0 and 1 that softened a join -- feathering, partial strength,
      whole-clip gradients and sigma-release all failed. It stays because
      partial pinning is a legitimate tool for other jobs; it is not the answer
      to a bad seam.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "This clip's fresh target latent."}),
            "previous_latent": ("LATENT", {"tooltip": "The previous clip's "
                                "SAMPLER output. No decode and re-encode: that "
                                "round trip is what climbed contrast down a "
                                "chain."}),
            "overlap_frames": (["39", "22", "5", "1", "0"], {"default": "39",
                               "tooltip": "Pixel frames of the previous clip to "
                                          "pin. Only these encode to distinct "
                                          "VAE runs; anything else lands at the "
                                          "wrong temporal phase and splices. "
                                          "39 -> 12 video steps and 65 audio "
                                          "steps, and is the smallest run on "
                                          "BOTH the 24fps and 40Hz grids. 0 "
                                          "passes through."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                         "step": 0.05,
                         "tooltip": "Leave at 1.0 to continue. Lower only "
                                    "half-preserves the prefix — it is not a "
                                    "seam control."}),
        }, "optional": {
            "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 256,
                                    "tooltip": "Half-cosine release over the "
                                               "last ticks of the AUDIO prefix. "
                                               "Video cuts from preserved to "
                                               "generated on a hard edge; a hard "
                                               "audio edge clicks. 8 = 0.2s at "
                                               "40 Hz. 0 = hard."}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "pinned_frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Copy the previous clip's tail into this clip's opening and "
                   "hold it, so the take continues instead of restarting.")

    def go(self, latent, previous_latent, overlap_frames, strength,
           audio_feather_ticks=8):
        import comfy.nested_tensor
        video, aud = av(latent["samples"])
        pv, pa = av(previous_latent["samples"])

        n = snap_context(int(overlap_frames))
        if n <= 0:
            return (latent, 0, "H3 LATENT PIN: overlap 0 — passed through")

        # 17n+5 pixel frames is 5n+2 latent steps, at both ends. Deriving the
        # step count from the run rather than rounding a ratio is what keeps the
        # copy phase-aligned.
        kv = 2 + 5 * ((n - 5) // 17) if n >= 5 else 1
        ka = int(round(n / 24.0 * 40))
        if kv >= int(video.shape[2]) or ka >= int(aud.shape[-1]):
            raise ValueError(
                f"H3 Latent Pin: a {n}-frame pin is {kv} video / {ka} audio "
                f"step(s), which does not fit inside a target of "
                f"{int(video.shape[2])} / {int(aud.shape[-1])}. Shorten the pin "
                f"or lengthen the clip.")
        if int(pv.shape[2]) < kv or int(pa.shape[-1]) < ka:
            raise ValueError(
                f"H3 Latent Pin: the previous clip has {int(pv.shape[2])} video "
                f"step(s) and the pin needs {kv}. It is shorter than the pin.")
        if tuple(pv.shape[3:]) != tuple(video.shape[3:]):
            raise ValueError(
                f"H3 Latent Pin: the previous clip is "
                f"{int(pv.shape[4]) * 16}x{int(pv.shape[3]) * 16} and this one "
                f"is {int(video.shape[4]) * 16}x{int(video.shape[3]) * 16}. The "
                f"pinned frames have to line up pixel for pixel with what "
                f"follows, so chained clips must share one resolution.")

        new_v, new_a = video.clone(), aud.clone()
        new_v[:, :, :kv] = pv[:, :, -kv:].to(new_v.device, new_v.dtype)
        new_a[..., :ka] = pa[..., -ka:].to(new_a.device, new_a.dtype)

        hold = 1.0 - float(strength)
        mask_v = torch.ones_like(video)
        mask_a = torch.ones_like(aud)
        mask_v[:, :, :kv] = hold
        mask_a[..., :ka] = hold
        f = max(0, min(int(audio_feather_ticks), ka))
        if f and strength > 0:
            # release the audio over its last ticks. Video cuts hard from
            # preserved to generated; audio does not survive that.
            ramp = torch.linspace(hold, 1.0, f + 2, device=mask_a.device,
                                  dtype=mask_a.dtype)[1:-1]
            ramp = 0.5 - 0.5 * torch.cos(ramp * 3.141592653589793)
            mask_a[..., ka - f:ka] = hold + (1.0 - hold) * ramp

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((new_v, new_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        info = (f"H3 LATENT PIN: {n} frame(s) = {kv} video / {ka} audio step(s) "
                f"copied and held at strength {strength:.2f}"
                + (f", audio released over the last {f} tick(s)" if f else "")
                + f"\n  {int(video.shape[2])} video step(s) in this clip, so "
                f"{100.0 * kv / int(video.shape[2]):.0f}% is reproduced")
        logging.info("H3LatentPin: %d frames = %d video / %d audio steps", n, kv, ka)
        return {"ui": {"h3char": [info]}, "result": (out, n, info)}


class H3LatentBracket:
    """Hold BOTH ends of a clip and generate the middle.

    THE QUESTION THIS EXISTS TO ANSWER
      Every seam we have measured had a pinned PAST and an open future. The
      failure is always the same shape -- "the model reproduces the handed-over
      prefix, then makes a fresh decision" -- and it has never been fixed:
      feathering, partial strength, whole-clip gradients and sigma-release all
      came back null over nine rounds, and three later attempts at sharing
      context between neighbours (drift control, the halo, the window margin)
      failed too.

      Every one of those gave the model MORE context about where it came from.
      None gave it anywhere to arrive. A held tail does: the last steps are
      stamped back at every denoise step, so the clip cannot end anywhere except
      where the source ends. The model is not free to make a fresh decision,
      because the decision is already made at both ends.

      That is structurally unlike everything in the failed pile, which is the
      whole reason to try it. It is NOT a prediction that it works.

    WHAT IT IS FOR BESIDES THAT
      Interior V2V. Feed a clip, keep its opening and its ending, regenerate
      what happens in between -- a different action, a different line, a
      different beat, with the shot arriving exactly where the rest of the edit
      needs it to. Front-and-tail extension is what a chain already does; this
      is the case chaining structurally cannot reach.

    WHAT IT DOES TO THE LATENT
      The source's first `head_frames` and last `tail_frames` are copied into
      the target in both streams and their noise mask set to 0. Nothing is added
      to the packed sequence -- no extra rows, no token cost. The middle is
      denoised normally.

    THE GRID, WHICH IS SIMPLER HERE THAN IN A CHAIN
      `H3LatentPin` needs legal runs (17n+5) because it moves content from the
      END of one clip to the START of another, and only 5j+2 steps land at the
      same phase after that move. Nothing MOVES here: head stays at the head,
      tail stays at the tail, so any count is phase-correct. What is worth
      having instead is whole VAE chunks, so a held region is a round number of
      them -- 5 latent steps cover 17 pixel frames wherever they start, so both
      widgets step by 17.

      Multiples of 51 also land on exact 40 Hz audio ticks (3 frames = 5 ticks).
      Anything else rounds by up to a tick at the boundary, which the feather
      covers.

    TWO TEMPORAL EDGES, NOT ONE
      Be clear-eyed: this has two of the edges that have never been softened.
      If the middle reads as bracketed by cuts, that is the known failure and
      not a surprise. The interesting outcome is the CONTENT between them --
      whether a model with a fixed destination drifts the way an open-ended one
      does.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "The target latent. Same shape and "
                        "length as the source — use H3 Match Source Clip."}),
            "source_latent": ("LATENT", {"tooltip": "The clip being edited. Its "
                               "head and tail are copied in and held. Prefer a "
                               "SAMPLER output or a single encode; a decode and "
                               "re-encode round trip is what climbed contrast "
                               "down a chain."}),
            "head_frames": ("INT", {"default": 34, "min": 0, "max": 3600,
                            "step": 17,
                            "tooltip": "Pixel frames held at the START. 5 latent "
                                       "steps cover 17 pixel frames, so this "
                                       "steps by 17 and a held region is always "
                                       "whole VAE chunks. Multiples of 51 also "
                                       "land on exact 40 Hz audio ticks."}),
            "tail_frames": ("INT", {"default": 34, "min": 0, "max": 3600,
                            "step": 17,
                            "tooltip": "Pixel frames held at the END. The half "
                                       "that has never been tested: every seam "
                                       "we have measured had a pinned past and "
                                       "an open future."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                         "step": 0.05,
                         "tooltip": "Leave at 1.0. Nine rounds of measurement "
                                    "found no value between 0 and 1 that "
                                    "softened a temporal join — it is not a "
                                    "seam control."}),
        }, "optional": {
            "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 256,
                                    "tooltip": "Half-cosine release out of the "
                                               "head and back into the tail. "
                                               "Video cuts hard from held to "
                                               "generated; a hard audio edge "
                                               "clicks. 8 = 0.2s at 40 Hz.\n\n"
                                               "Above 0 this is a FRACTIONAL "
                                               "mask, and ComfyUI 0.35 scales "
                                               "the model's output by the mask "
                                               "— that interaction is open. 0 "
                                               "keeps the mask binary."}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "generated_frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Hold the opening and the ending of a clip, generate the "
                   "middle. The model gets a destination, not just a past.")

    def go(self, latent, source_latent, head_frames, tail_frames, strength,
           audio_feather_ticks=8):
        import comfy.nested_tensor
        video, aud = av(latent["samples"])
        sv, sa = av(source_latent["samples"])

        t_v, t_a = int(video.shape[2]), int(aud.shape[-1])
        if tuple(sv.shape[2:]) != tuple(video.shape[2:]):
            raise ValueError(
                f"H3 Latent Bracket: the source is {int(sv.shape[2])} step(s) at "
                f"{int(sv.shape[4]) * 16}x{int(sv.shape[3]) * 16} and the target "
                f"is {t_v} at {int(video.shape[4]) * 16}x"
                f"{int(video.shape[3]) * 16}. Held frames are copied in place, "
                f"so the two have to be the same clip shape — wire H3 Match "
                f"Source Clip into the conditioning node.")

        # 5 latent steps cover 17 pixel frames wherever they start, so a whole
        # number of chunks at either end needs no phase arithmetic at all.
        hf = max(0, int(head_frames) // 17 * 17)
        tf = max(0, int(tail_frames) // 17 * 17)
        hv, tv = hf // 17 * 5, tf // 17 * 5
        ha, ta = audio_t(hf), audio_t(tf)

        if hv + tv >= t_v:
            raise ValueError(
                f"H3 Latent Bracket: holding {hf} + {tf} frames is {hv} + {tv} "
                f"of {t_v} video step(s), which leaves nothing to generate. "
                f"Shorten the brackets or lengthen the clip.")
        if ha + ta >= t_a:
            raise ValueError(
                f"H3 Latent Bracket: the audio brackets are {ha} + {ta} of "
                f"{t_a} tick(s), leaving no middle.")
        total_px = sum(frame_groups(t_v))
        new_v, new_a = video.clone(), aud.clone()
        hold = 1.0 - float(strength)
        mask_v = torch.ones_like(video)
        mask_a = torch.ones_like(aud)

        if hv == 0 and tv == 0:
            # PASS THROUGH WITH A MASK, NOT WITHOUT ONE. An all-ones mask is a
            # no-op for the sampler and costs nothing, but its ABSENCE is not:
            # H3ChunkLatentContext shipped a bare passthrough and downstream
            # nodes that read `noise_mask` refused the latent outright. Same
            # lesson, so make the same shape either way.
            out = dict(latent)
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
            msg = (f"H3 LATENT BRACKET: nothing held — {int(head_frames)} and "
                   f"{int(tail_frames)} both fall below one VAE chunk (17 "
                   f"frames), so the whole clip is generated.")
            return {"ui": {"h3char": [msg]}, "result": (out, int(total_px), msg)}

        if hv:
            new_v[:, :, :hv] = sv[:, :, :hv].to(new_v.device, new_v.dtype)
            mask_v[:, :, :hv] = hold
        if ha:
            new_a[..., :ha] = sa[..., :ha].to(new_a.device, new_a.dtype)
            mask_a[..., :ha] = hold
        # NEGATIVE INDEXING WOULD BE A SILENT NO-OP AT 0. `x[..., -0:]` is the
        # WHOLE tensor, not an empty slice, so a zero-length tail would hold the
        # entire clip and generate nothing. Guarded by the `if`, and written with
        # explicit bounds so the guard is not the only thing standing between
        # this and a render that ignores the prompt entirely.
        if tv:
            new_v[:, :, t_v - tv:] = sv[:, :, t_v - tv:].to(new_v.device,
                                                            new_v.dtype)
            mask_v[:, :, t_v - tv:] = hold
        if ta:
            new_a[..., t_a - ta:] = sa[..., t_a - ta:].to(new_a.device,
                                                          new_a.dtype)
            mask_a[..., t_a - ta:] = hold

        f = int(audio_feather_ticks)
        if f and strength > 0:
            def _smoothstep(n):
                """n points rising 0 -> 1 on a half cosine, endpoints excluded."""
                r = torch.linspace(0.0, 1.0, n + 2, device=mask_a.device,
                                   dtype=mask_a.dtype)[1:-1]
                return 0.5 - 0.5 * torch.cos(r * 3.141592653589793)

            n = min(f, ha)
            if n:
                # leaving the head: held -> generated
                mask_a[..., ha - n:ha] = hold + (1.0 - hold) * _smoothstep(n)
            n = min(f, ta)
            if n:
                # entering the tail: generated -> held, the mirror image
                mask_a[..., t_a - ta:t_a - ta + n] = (
                    1.0 - (1.0 - hold) * _smoothstep(n))

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((new_v, new_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))

        gen_v = t_v - hv - tv
        gen_px = max(0, total_px - hf - tf)
        info = "\n".join([
            f"H3 LATENT BRACKET: holding {hf} frame(s) at the head and {tf} at "
            f"the tail",
            f"  video  {hv} + {tv} of {t_v} step(s) held, {gen_v} generated",
            f"  audio  {ha} + {ta} of {t_a} tick(s) held"
            + (f", feathered over {min(f, ha or f)} tick(s) each side" if f else
               " (hard edges)"),
            f"  generated span: frames {hf}-{hf + gen_px} of {total_px}",
            "  BOTH edges are temporal mask edges, and no setting has ever "
            "softened one. What is new is that the model has a destination.",
        ])
        logging.info("H3LatentBracket: head %df (%d/%d) tail %df (%d/%d), "
                     "generating %d video step(s)",
                     hf, hv, ha, tf, tv, ta, gen_v)
        return {"ui": {"h3char": [info]}, "result": (out, int(gen_px), info)}


def _frames_of(latent_t):
    """Pixel frames in a clip of `latent_t` video steps. Inverse of core's
    `video_latent_t`: 17n+5 frames <-> 5n+2 steps."""
    return max(5, (int(latent_t) - 2) // 5 * 17 + 5)


def _latent_t(frames):
    """Core's `video_latent_t`, repeated so a node can SIZE a canvas rather than
    be handed one. Kept next to its inverse so the pair cannot drift."""
    return 2 if int(frames) <= 5 else (int(frames) - 5) // 17 * 5 + 2


# WHERE A CUT IS ALLOWED TO LAND, which is finer than the 17-frame chunk.
#
# A group of 5 latent steps covers 17 pixel frames, but not evenly: the steps
# cover 1, 4, 4, 4, 4. So the boundaries inside a group sit at frame offsets
# 0, 1, 5, 9, 13 -- FIVE places to cut per chunk, about one every 4 frames,
# rather than one every 17.
#
# The 17-frame rule belongs to the SHIFT, not the cut. Sliding a tail by a whole
# group keeps every step's k%5, and 5 consecutive steps cover 17 frames wherever
# they start, so the shift is a multiple of 17 no matter which boundary the cut
# is on. Conflating the two cost a factor of four in precision for nothing.
#
# Derived from the canonical table so a change to the grid propagates here.
_STEP_STARTS = tuple(sum(FRAME_PER_TOKEN[:k]) for k in range(len(FRAME_PER_TOKEN)))
_GROUP = sum(FRAME_PER_TOKEN)


def frames_at_step(step):
    """Pixel frames lying before latent step `step`."""
    q, r = divmod(max(0, int(step)), len(FRAME_PER_TOKEN))
    return q * _GROUP + _STEP_STARTS[r]


def step_at_frame(frames, direction="down"):
    """The latent step whose boundary is at or below (or above) `frames`.

    A latent step is the atom -- you cannot hold half of one -- so every cut
    lands on one of these. `down` never regenerates less than asked; `up` never
    holds more than asked.
    """
    g, r = divmod(max(0, int(frames)), _GROUP)
    n = len(FRAME_PER_TOKEN)
    if direction == "down":
        return g * n + max(k for k, o in enumerate(_STEP_STARTS) if o <= r)
    for k, o in enumerate(_STEP_STARTS):
        if o >= r:
            return g * n + k
    return (g + 1) * n


class H3LatentInsert:
    """Cut an existing clip at a frame and generate NEW frames into the gap.

    THE DIFFERENCE FROM H3 LATENT BRACKET, WHICH IS THE WHOLE POINT
      The bracket holds both ends and regenerates the middle AT THE SAME LENGTH:
      whatever sat between the brackets is overwritten and the take is exactly as
      long as it was. This makes the take LONGER. The source is cut at
      `split_frame`, everything before it stays at the head, everything after it
      slides down by `insert_frames`, and the span opened up between them is
      generated. Nothing of the source is lost unless you ask for it.

      `split_frame = 0` puts the new material in FRONT of the clip; `split_frame`
      at or past the last frame puts it after; anything between is an interior
      insert. One node covers head, middle and tail extension.

    WHY THE MOVE IS SAFE HERE AND IS NOT IN A CHAIN
      [[h3-motion-context]]: `FRAME_PER_TOKEN = (1,4,4,4,4)`, so coverage is
      POSITIONAL -- latent step k covers 1 pixel frame when k%5==0 and 4
      otherwise. Slide content by an arbitrary number of steps and every step
      lands on a different coverage than it was encoded for, which is the
      measured failure behind "never slice a previous latent".

      `insert_frames` is therefore a multiple of 17 pixel frames = 5 latent
      steps, so the tail moves by a whole number of GROUPS and every step keeps
      its k%5. That is not a convention to be polite about the grid; it is the
      reason this is a copy and not a resample.

      THE CUT IS NOT ON THAT GRID, and an earlier version of this node wrongly
      put it there. 5 consecutive steps cover 17 frames WHEREVER THEY START, so
      the shift is a whole group no matter which boundary the tail begins on.
      The cut only has to land on a step, and steps start at frame offsets
      0, 1, 5, 9, 13 within each group -- five places per chunk, about one every
      4 frames. Conflating the shift's grid with the cut's cost a factor of four
      in precision for nothing.

      The tail is written anchored to the END of the target, not by computing the
      shift -- same result, and it cannot drift by a step if the arithmetic above
      is ever wrong.

    CHANGING THE SHOT AS WELL AS EXTENDING IT
      `blend_before` / `blend_after` widen the generated span INTO the source on
      either side of the cut, so the new material does not have to butt straight
      against untouched footage. At `insert_frames = 0` they are the bracket:
      a hole in the middle of a clip of unchanged length. Above 0 you get both --
      new frames, and a rewritten run-up and run-out to carry into them.

    IT MAKES ITS OWN CANVAS, AND THAT IS THE POINT
      The first version took the blank latent from the conditioning node and
      demanded it already be `source + insert` frames long, which meant typing
      the same number into two nodes and keeping them in sync. That is a trap,
      not a feature: it failed the first time it was used.

      The length lives ONLY in the canvas. `MiniMaxH3ReferenceToVideo` sizes an
      empty latent from its `length` widget and hands the conditioning the
      references and the text -- read its `execute`: `frame_count` is used for
      nothing else except trimming reference VIDEOS. So this node knows the
      answer already (source length + insert) and allocates the canvas itself.

      `latent` is therefore optional. Wire it and it is used when it is already
      the right size, so anything upstream put in it survives; wire something of
      the wrong length and it is replaced, reported, not refused. Leave it
      unwired and the canvas is built from the source's own shape.

      The conditioning node's `length` still wants to be right, because a
      reference video longer than it gets trimmed to it -- but getting it wrong
      can no longer break the render.

    THE AUDIO GRID
      Audio is a flat 40 Hz axis, so head audio is anchored at 0 and tail audio
      at the end and the gap between them is exactly the insert's duration -- no
      rounding accumulates. 3 frames = 5 ticks, so multiples of 51 frames land on
      exact ticks; anything else is off by up to one, which the feather covers.

    STILL TWO TEMPORAL EDGES
      Same honest warning as the bracket: both ends of the generated span are
      temporal mask edges, and nothing has ever softened one. What this buys is
      that the model has a fixed destination to arrive at, and that the footage
      on both sides of it survives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_latent": ("LATENT", {"tooltip": "The clip being extended. "
                               "Prefer a SAMPLER output or a single encode; a "
                               "decode and re-encode round trip is what climbed "
                               "contrast down a chain."}),
            "split_frame": ("INT", {"default": 0, "min": 0, "max": 3600,
                            "tooltip": "Where the source is cut, in its OWN "
                                       "frames. 0 inserts in front of the clip; "
                                       "at or past the last frame appends after "
                                       "it; anything between is an interior "
                                       "insert.\n\nType any frame. It snaps DOWN "
                                       "to the nearest latent step boundary — "
                                       "those sit at offsets 0, 1, 5, 9 and 13 in "
                                       "every 17 frames, so about one every 4 "
                                       "frames — and the info says where it "
                                       "landed. A step is the atom; holding half "
                                       "of one is not a thing."}),
            "insert_frames": ("INT", {"default": 68, "min": 0, "max": 3600,
                              "step": 17,
                              "tooltip": "How many NEW frames open up at the cut, "
                                         "and so how much longer the take gets. "
                                         "The canvas is sized from this — nothing "
                                         "upstream has to be told. 0 turns the "
                                         "node into H3 Latent Bracket."}),
            "blend_before": ("INT", {"default": 0, "min": 0, "max": 3600,
                             "tooltip": "Source frames BEFORE the cut that are "
                                        "regenerated along with the insert, so "
                                        "the new material has a run-up written "
                                        "for it instead of butting against "
                                        "untouched footage. Snaps to a step "
                                        "boundary the generous way — you never "
                                        "get less blend than you asked for."}),
            "blend_after": ("INT", {"default": 0, "min": 0, "max": 3600,
                            "tooltip": "The same on the far side of the cut — "
                                       "source frames rewritten as the run-out."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                         "step": 0.05,
                         "tooltip": "Leave at 1.0. Nine rounds of measurement "
                                    "found no value between 0 and 1 that "
                                    "softened a temporal join — it is not a "
                                    "seam control."}),
        }, "optional": {
            "latent": ("LATENT", {"tooltip": "OPTIONAL. The canvas is sized from "
                        "the source and insert_frames, so this is not needed. "
                        "Wire the conditioning node's latent here and it is used "
                        "when it already happens to be the right length, so "
                        "anything upstream put in it survives; at any other "
                        "length it is replaced and the info says so."}),
            "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 256,
                                    "tooltip": "Half-cosine release out of the "
                                               "held head and back into the held "
                                               "tail. Video cuts hard; a hard "
                                               "audio edge clicks. 8 = 0.2s at "
                                               "40 Hz.\n\nAbove 0 this is a "
                                               "FRACTIONAL mask, and ComfyUI 0.35 "
                                               "scales the model's output by the "
                                               "mask — that interaction is open. "
                                               "0 keeps the mask binary."}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "generated_frames", "total_frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = ("Cut a clip at a frame and generate new frames into the gap. "
                   "The take gets longer; the footage either side survives.")

    def go(self, source_latent, split_frame, insert_frames, blend_before,
           blend_after, strength, latent=None, audio_feather_ticks=8):
        import comfy.nested_tensor
        sv, sa = av(source_latent["samples"])
        s_v, s_a = int(sv.shape[2]), int(sa.shape[-1])
        s_px = _frames_of(s_v)
        ins = max(0, int(insert_frames) // 17 * 17)
        t_px = s_px + ins
        t_v, t_a = _latent_t(t_px), audio_t(t_px)

        # THE CANVAS IS SIZED HERE, not upstream. The length lives only in the
        # latent -- the conditioning node's `length` widget sizes an empty one
        # and is otherwise used for nothing but trimming reference videos -- so
        # this node knows the answer already and making the graph agree with it
        # by hand was a trap that failed the first time it was used.
        canvas = None
        if latent is not None:
            lv, la = av(latent["samples"])
            if (int(lv.shape[2]), int(la.shape[-1])) == (t_v, t_a) and \
                    tuple(lv.shape[3:]) == tuple(sv.shape[3:]):
                canvas = (lv, la)
        note = ""
        if canvas is None:
            video = torch.zeros((sv.shape[0], sv.shape[1], t_v, *sv.shape[3:]),
                                device=sv.device, dtype=sv.dtype)
            aud = torch.zeros((sa.shape[0], sa.shape[1], sa.shape[2], t_a),
                              device=sa.device, dtype=sa.dtype)
            if latent is not None:
                given = _frames_of(int(av(latent["samples"])[0].shape[2]))
                note = (f"  the wired latent is {given} frame(s) and this needs "
                        f"{t_px}, so the canvas was built here instead. Set the "
                        f"conditioning node's `length` to {t_px} if it carries "
                        f"reference VIDEOS, which get trimmed to it.")
        else:
            video, aud = canvas

        # THE CUT LANDS ON A LATENT STEP, not on a 17-frame chunk. Five boundaries
        # per chunk, so about one every 4 frames — the 17 belongs to the SHIFT,
        # which is a whole group wherever the cut is. Clamped to the source first
        # so a split past the end means "the end" without a special case.
        def _at(frames, direction="down"):
            return min(s_v, step_at_frame(min(max(0, frames), s_px), direction))

        # Before the cut we snap DOWN and after it UP, so asking for a blend
        # always gets at least what you asked for rather than less.
        want = min(max(0, int(split_frame)), s_px)
        cut_step = _at(want)
        cut = frames_at_step(cut_step)
        hv = _at(cut - max(0, int(blend_before)))
        from_step = _at(cut + max(0, int(blend_after)), "up")
        tail_v = s_v - from_step

        keep_to, keep_from = frames_at_step(hv), frames_at_step(from_step)
        ha = audio_t(keep_to)
        tail_a = s_a - audio_t(keep_from)

        gen_v = t_v - hv - tail_v
        gen_px = t_px - keep_to - (s_px - keep_from)
        if gen_v <= 0:
            raise ValueError(
                f"H3 Latent Insert: holding frames 0-{keep_to} and "
                f"{keep_from}-{s_px} of the source leaves {gen_v} video step(s) "
                f"to generate. Insert more frames, or widen blend_before / "
                f"blend_after.")

        new_v, new_a = video.clone(), aud.clone()
        hold = 1.0 - float(strength)
        mask_v = torch.ones_like(video)
        mask_a = torch.ones_like(aud)

        # NEGATIVE INDEXING WOULD BE A SILENT NO-OP AT 0 — `x[..., -0:]` is the
        # whole tensor, so a zero-length tail would hold the entire clip and
        # generate nothing. Explicit bounds, guarded, same as the bracket.
        if hv:
            new_v[:, :, :hv] = sv[:, :, :hv].to(new_v.device, new_v.dtype)
            mask_v[:, :, :hv] = hold
        if ha:
            new_a[..., :ha] = sa[..., :ha].to(new_a.device, new_a.dtype)
            mask_a[..., :ha] = hold
        if tail_v:
            new_v[:, :, t_v - tail_v:] = sv[:, :, s_v - tail_v:].to(new_v.device,
                                                                    new_v.dtype)
            mask_v[:, :, t_v - tail_v:] = hold
        if tail_a:
            new_a[..., t_a - tail_a:] = sa[..., s_a - tail_a:].to(new_a.device,
                                                                  new_a.dtype)
            mask_a[..., t_a - tail_a:] = hold

        f = int(audio_feather_ticks)
        if f and strength > 0:
            def _smoothstep(n):
                r = torch.linspace(0.0, 1.0, n + 2, device=mask_a.device,
                                   dtype=mask_a.dtype)[1:-1]
                return 0.5 - 0.5 * torch.cos(r * 3.141592653589793)

            n = min(f, ha)
            if n:
                mask_a[..., ha - n:ha] = hold + (1.0 - hold) * _smoothstep(n)
            n = min(f, tail_a)
            if n:
                mask_a[..., t_a - tail_a:t_a - tail_a + n] = (
                    1.0 - (1.0 - hold) * _smoothstep(n))

        out = dict(latent) if latent is not None else {}
        out["samples"] = comfy.nested_tensor.NestedTensor((new_v, new_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))

        where = ("in front of the clip" if cut == 0 else
                 "after the clip" if cut >= s_px else f"at frame {cut}"
                 + (f" (asked for {want}, nearest step boundary below)"
                    if cut != want else ""))
        info = "\n".join([
            f"H3 LATENT INSERT: {ins} new frame(s) {where} — {s_px} -> {t_px} "
            f"({describe(t_px)})",
            f"  held    source 0-{keep_to} and {keep_from}-{s_px}"
            + (f", regenerating {cut - keep_to} frame(s) of run-up" if keep_to < cut
               else "")
            + (f" and {keep_from - cut} of run-out" if keep_from > cut else ""),
            f"  video   {hv} + {tail_v} of {t_v} step(s) held, {gen_v} generated",
            f"  audio   {ha} + {tail_a} of {t_a} tick(s) held"
            + (f", feathered over {f} tick(s) each side" if f else " (hard edges)"),
            f"  new span: frames {keep_to}-{keep_to + gen_px} of {t_px}",
            "  BOTH edges of that span are temporal mask edges and no setting has "
            "ever softened one. What is new is that the footage either side of it "
            "survives.",
        ] + ([note] if note else []))
        logging.info("H3LatentInsert: cut %d, +%d frames (%d->%d), holding "
                     "%d+%d of %d video step(s)",
                     cut, ins, s_px, t_px, hv, tail_v, t_v)
        return {"ui": {"h3char": [info]},
                "result": (out, int(gen_px), int(t_px), info)}


class H3MatchSource:
    """Derive width / height / length from a source clip, so masking lines up.

    H3MaskInpaint pins the source's own pixels outside the mask, so the generated
    latent must have exactly the source's shape. Typing the numbers by hand fails
    the moment a clip is not the resolution you assumed:

        source video encodes to (12, 22, 40) but the latent is (12, 70, 40)

    which is a 640x352 clip against a node still set to 640x1120.

    Frame count is snapped DOWN to the 17n+5 grid the video VAE requires, and the
    images are trimmed to match, so the mask and the source stay frame-aligned.
    Trimming the tail is safe; padding would invent frames the mask does not cover.

    HOW IT CONFORMS IS YOUR CHOICE. An earlier version only ever cropped to /32,
    on the argument that an inpaint keeps most of the source verbatim so
    resampling spends sharpness on pixels that were going to survive. That
    argument is real but it is not the whole picture: crop-to-/32 cannot make a
    1920x1080 source into anything H3 will render. 1920x1056 is 2 MP against a
    768x1344 cap, so real footage needs a downscale, and once you are resampling
    anyway the only question left is HOW. So:

      crop to /32   trim to the nearest multiple of 32. No resampling at all, and
                    no scaling — only useful when the source is already close to
                    a canvas H3 can render.
      fill          centre-crop to the target's aspect, then scale. No distortion,
                    no bars, loses the edges that do not fit. The usual choice.
      stretch       scale straight to the target. Keeps the whole frame and
                    distorts it. What ImageResizeKJv2's 'resize' does.
      pad           scale to fit inside the target, bars for the remainder. Keeps
                    everything undistorted, but the model will try to generate
                    into the bars, so expect to crop them off afterwards.
      none          pass through untouched; errors if the clip is not already
                    legal. For when something upstream already conformed it.
    """

    MODES = ["crop to /32", "fill", "stretch", "pad", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The source clip."}),
            "mode": (cls.MODES, {"default": "fill",
                     "tooltip": "How to conform the clip to the target. 'fill' keeps "
                                "the aspect and loses the edges; 'stretch' keeps "
                                "everything and distorts; 'pad' keeps everything and "
                                "adds bars the model will try to paint into; "
                                "'crop to /32' does not scale at all."}),
            "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                             "tooltip": "0 = derive from the source. Set both to "
                                        "downscale a big clip to something H3 can "
                                        "render — the short edge wants to be around "
                                        "768 and the area cap is 768x1344."}),
            "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "target_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0,
                                  "step": 0.05,
                                  "tooltip": "0 = off. Sets a pixel BUDGET and keeps "
                                             "the source's shape — the dimensions are "
                                             "derived from the source aspect at this "
                                             "area, so there is no aspect conflict and "
                                             "the mode does not matter. H3's cap is "
                                             "1.03 MP (768x1344); 0.72 is 640x1120. "
                                             "Ignored if target_width/height are set, "
                                             "because those already fix the shape."}),
        }, "optional": {
            "av_aligned": ("BOOLEAN", {"default": False,
                           "tooltip": "Also trim to a run that lands exactly on the 40 Hz "
                                      "audio grid. Costs up to 51 more frames, so it is "
                                      "off by default — worth it when this clip is a link "
                                      "in a chain, not when you are masking a one-off."}),
            "mask": ("MASK", {"tooltip": "Optional — conformed and trimmed identically, "
                                         "so it stays pixel-aligned with the frames."}),
            "mask_2": ("MASK", {"tooltip": "A second mask riding the same conform and "
                                           "trim. For anything derived from the same "
                                           "frames that has to stay aligned with them — "
                                           "a forget mask, an occluder mask. Rebuilding "
                                           "this path from stock resize nodes looks "
                                           "equivalent and is not: it misses the frame "
                                           "trim, so the mask ends up longer than the "
                                           "clip."}),
        }}

    # mask_2 sits next to mask, where it reads. That is a deliberate break: slot
    # indices are what saved workflows store, so everything after it shifted by one
    # and graphs built before 2026-08-21 need their width/height/length/info links
    # remade. The shipped workflows were remapped; a hand-built one will show the
    # wrong wires rather than fail loudly, so check them.
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "mask", "mask_2", "width", "height", "length", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/mask"
    DESCRIPTION = ("Conform a source clip to a canvas H3 can render and report the "
                   "width, height and legal frame count. Wire all three into the H3 "
                   "conditioning node so an inpaint can never be shape-mismatched.")

    def go(self, images, mode="fill", target_width=0, target_height=0,
           target_megapixels=0.0, av_aligned=False, mask=None, mask_2=None):
        import math
        import torch.nn.functional as Fn
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        notes = []

        # every mask travels as a LIST through one code path. Threading three
        # variables through five mode branches is how the second one quietly stops
        # matching the first.
        masks = [m if m is None or m.dim() == 3 else m.unsqueeze(0)
                 for m in (mask, mask_2)]

        def crop_all(ms, y0, ch, x0, cw):
            return [None if m is None else m[..., y0:y0 + ch, x0:x0 + cw] for m in ms]

        def scale(img, ms, tw, th):
            img = Fn.interpolate(img.movedim(-1, 1), size=(th, tw), mode="bicubic",
                                 align_corners=False,
                                 antialias=True).clamp(0, 1).movedim(1, -1)
            ms = [None if m is None else
                  Fn.interpolate(m.unsqueeze(1), size=(th, tw),
                                 mode="nearest").squeeze(1) for m in ms]
            return img, ms

        tw = (int(target_width) // 32) * 32
        th = (int(target_height) // 32) * 32
        want_target = tw >= 32 and th >= 32

        if float(target_megapixels) > 0:
            if want_target:
                notes.append("target_megapixels ignored — width/height already fix "
                             "the canvas")
            else:
                # keep the source's shape and hit the area budget. Both axes still
                # land on 32, so the ratio moves a little and the area is close
                # rather than exact.
                area = float(target_megapixels) * 1e6
                ar = w / float(h)
                tw = max(32, int(round(math.sqrt(area * ar) / 32)) * 32)
                th = max(32, int(round(math.sqrt(area / ar) / 32)) * 32)
                want_target = True
                notes.append(f"{target_megapixels:.2f} MP at the source's "
                             f"{ar:.2f}:1 -> {tw}x{th}")
                if mode in ("fill", "stretch", "pad"):
                    # the derived canvas already matches the source's shape, so
                    # there is no conflict for the mode to resolve
                    mode = "fill"

        if mode == "none":
            if w % 32 or h % 32:
                raise ValueError(
                    f"mode 'none' but the clip is {w}x{h}, which is not a multiple of "
                    f"32. Pick another mode, or conform it upstream.")
            cw, ch = w, h
        elif mode == "crop to /32" or not want_target:
            if want_target:
                notes.append("target ignored — 'crop to /32' does not scale")
            try:
                x0, y0, cw, ch = crop_to_multiple(w, h, 32)
            except ValueError:
                raise ValueError(f"source is {w}x{h}; H3 needs at least 32x32 after "
                                 f"cropping to a multiple of 32.")
            if (cw, ch) != (w, h):
                images = images[:, y0:y0 + ch, x0:x0 + cw, :]
                masks = crop_all(masks, y0, ch, x0, cw)
                notes.append(f"centre-cropped {w}x{h} -> {cw}x{ch} (no resample)")
            if mode != "crop to /32" and not want_target:
                notes.append(f"no target set, so '{mode}' fell back to a /32 crop")
        elif mode == "fill":
            x0, y0, cw0, ch0 = cover_crop(w, h, tw, th)
            if (cw0, ch0) != (w, h):
                images = images[:, y0:y0 + ch0, x0:x0 + cw0, :]
                masks = crop_all(masks, y0, ch0, x0, cw0)
                notes.append(f"cropped to {tw}:{th} aspect ({cw0}x{ch0})")
            images, masks = scale(images, masks, tw, th)
            cw, ch = tw, th
            notes.append(f"scaled to {tw}x{th}")
        elif mode == "stretch":
            images, masks = scale(images, masks, tw, th)
            cw, ch = tw, th
            ar_s, ar_t = w / max(1, h), tw / max(1, th)
            notes.append(f"stretched {w}x{h} -> {tw}x{th}")
            if max(ar_s, ar_t) / min(ar_s, ar_t) > 1.05:
                notes.append(f"WARNING: {ar_s:.2f}:1 into {ar_t:.2f}:1 distorts the "
                             f"picture — the model will be asked to match a squashed "
                             f"body. 'fill' avoids this")
        elif mode == "pad":
            s = min(tw / w, th / h)
            iw, ih = max(32, int(round(w * s))), max(32, int(round(h * s)))
            images, masks = scale(images, masks, iw, ih)
            px, py = (tw - iw) // 2, (th - ih) // 2
            pad = (px, tw - iw - px, py, th - ih - py)
            images = Fn.pad(images.movedim(-1, 1), pad).movedim(1, -1)
            masks = [None if m is None else Fn.pad(m, pad) for m in masks]
            cw, ch = tw, th
            notes.append(f"scaled to {iw}x{ih} and padded to {tw}x{th}")
            notes.append("WARNING: the model generates into the bars — crop them off "
                         "after, or use 'fill'")
        else:
            raise ValueError(f"unknown mode {mode!r}")

        length = n
        while length > 5 and length % 17 != 5:
            length -= 1
        if length < 5:
            raise ValueError(f"only {n} frame(s); the smallest legal clip is 5.")
        if av_aligned:
            aligned = snap_av_aligned(length, "down")
            if aligned <= length:
                length = aligned
            else:
                notes.append("too short for any AV-aligned run; left on the video grid")
        if length != n:
            notes.append(f"trimmed {n - length} frame(s)")
        images = images[:length]
        # THE TRIM is why this belongs on the node rather than being rebuilt from
        # stock resize nodes: those scale but do not shorten, so a second mask ends
        # up longer than the clip and H3 Mask Inpaint rejects it.
        masks = [None if m is None else m[:length] for m in masks]

        mp = cw * ch / 1e6
        info = f"{cw}x{ch} ({mp:.2f} MP), " + describe(length)
        if notes:
            info += " — " + "; ".join(notes)
        if mp > 1.05:
            info += ("\nWARNING: past the 768x1344 area cap. Set target_width/height "
                     "to something H3 renders — a bigger canvas measured no better and "
                     "costs superlinearly.")
        # an unconnected mask output is an empty one, so downstream nodes get a
        # well-formed tensor rather than None
        blank = torch.zeros(length, ch, cw)
        m1, m2 = [blank if m is None else m for m in masks]
        if masks[1] is not None:
            info += " | 2 masks conformed together"
        return {"ui": {"h3char": [info]},
                "result": (images, m1, m2, cw, ch, length, info)}



class H3MaskStabilize:
    """Settle a mask that flickers frame to frame, without growing it.

    Segmentation is re-run per frame, so it drops and re-acquires whatever is
    hardest — fast, small, self-occluding things. Hands, mostly. The mask then
    switches on and off across consecutive frames, and every switch is a change in
    what the sampler is allowed to touch there.

    That matters more than it looks. A latent frame covers up to four pixel frames
    and the mask is UNIONED over them, so a hand that drops out for one frame turns
    a solid latent cell into a partial one, and the free/pinned boundary moves. The
    model gets a different instruction at that spot every latent step.

    WHY NOT A MEDIAN FILTER
      The obvious tool, and wrong here. A median across frames asks "is this pixel
      usually masked", which for a MOVING subject is no — the hand is only at any
      given pixel briefly. A median erases fast motion, which is exactly what needs
      keeping.

    So this uses morphology along the time axis instead, which is about GAPS rather
    than averages:

      remove_blips   temporal opening. Drops anything that appears for fewer than
                     this many frames — spurious detections that flash on and off.
      fill_gaps      temporal closing. Fills dropouts shorter than this, so a hand
                     that vanishes for a frame or two stays masked through it.

    Both preserve the mask's overall extent: a region genuinely present keeps its
    shape and its path. Only the holes in time close up.

    Order is opening then closing, so a blip is not first widened and then treated
    as real.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "fill_gaps": ("INT", {"default": 2, "min": 0, "max": 16,
                          "tooltip": "Frames. Dropouts shorter than this are filled. "
                                     "2 covers the common case of a hand lost for a "
                                     "frame or two. Too high and a genuine "
                                     "disappearance gets papered over."}),
            "remove_blips": ("INT", {"default": 1, "min": 0, "max": 16,
                             "tooltip": "Frames. Detections lasting fewer than this "
                                        "are dropped. 1 clears single-frame noise; "
                                        "raise it only if you see the mask flashing "
                                        "onto things that are not the subject."}),
        }}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Fill frame-to-frame dropouts in a mask and drop momentary false "
                   "detections, without growing the mask or blurring its motion. For "
                   "hands and anything else segmentation loses track of.")

    def go(self, mask, fill_gaps, remove_blips):
        import torch.nn.functional as Fn
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        m = m.float()
        n = m.shape[0]
        before = float(m.mean())
        x = m.unsqueeze(0).unsqueeze(0)                      # [1,1,N,H,W]

        def dilate_t(t, r):
            return Fn.max_pool3d(t, (2 * r + 1, 1, 1), stride=1, padding=(r, 0, 0))

        def erode_t(t, r):
            return -Fn.max_pool3d(-t, (2 * r + 1, 1, 1), stride=1, padding=(r, 0, 0))

        # opening first: a one-frame blip should not be widened and then kept
        if remove_blips > 0:
            r = int(remove_blips)
            x = dilate_t(erode_t(x, r), r)
        if fill_gaps > 0:
            r = int(fill_gaps)
            x = erode_t(dilate_t(x, r), r)

        out = x[0, 0].clamp(0, 1)
        after = float(out.mean())

        # how unsteady was it? count per-frame coverage swings before and after
        def jitter(t):
            per = t.flatten(1).mean(dim=1)
            return float((per[1:] - per[:-1]).abs().mean()) if t.shape[0] > 1 else 0.0

        j0, j1 = jitter(m), jitter(out)
        info = (f"{n} frames | coverage {before * 100:.2f}% -> {after * 100:.2f}% | "
                f"frame-to-frame jitter {j0 * 100:.3f}% -> {j1 * 100:.3f}%")
        if j0 > 0 and j1 < j0 * 0.9:
            info += f" ({(1 - j1 / j0) * 100:.0f}% steadier)"
        elif j0 > 0:
            info += " (little change — the flicker may be motion, not dropout)"
        logging.info("H3MaskStabilize: %s", info)
        return {"ui": {"h3char": [info]}, "result": (out, info)}


NODE_CLASS_MAPPINGS = {
    "H3MaskInpaint": H3MaskInpaint,
    "H3LatentPin": H3LatentPin,
    "H3LatentBracket": H3LatentBracket,
    "H3LatentInsert": H3LatentInsert,
    "H3MatchSource": H3MatchSource,
    "H3MaskStabilize": H3MaskStabilize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MaskInpaint": "H3 Mask Inpaint (region replace)",
    "H3LatentPin": "H3 Latent Pin (cuts — read description)",
    "H3LatentBracket": "H3 Latent Bracket (hold both ends)",
    "H3LatentInsert": "H3 Latent Insert (extend at a cut)",
    "H3MatchSource": "H3 Match Source Clip",
    "H3MaskStabilize": "H3 Stabilize Mask (temporal)",
}
