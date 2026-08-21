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
from .geometry import cover_crop, crop_to_multiple
from .timing import (FPS, align_frames, av_aligned_runs_through, describe,
                     frame_groups, is_av_aligned, snap_av_aligned)

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
           audio_vae=None, source_audio=None, forget_mask=None, forget_strength=1.0,
           sigmas=None):
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
            m = Fn.avg_pool3d(m, kernel_size=(1, k, k), stride=1, padding=(0, r, r))
            m = m.clamp(0, 1)

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
            # forgetting outside the regenerate region is meaningless — those cells
            # are re-pinned to the source every step and would overwrite it anyway
            f = f * mask_v

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
        video, aud = av(latent["samples"])
        pv, pa = av(previous_latent["samples"])

        kv = min(video_latent_t(max(5, overlap_frames)), video.shape[2] - 1)
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

    # mask_2 is APPENDED, not inserted next to mask. Slot indices are what saved
    # workflows store, so inserting in the middle silently rewires every graph that
    # already uses this node.
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING", "MASK")
    RETURN_NAMES = ("images", "mask", "width", "height", "length", "info", "mask_2")
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
                "result": (images, m1, cw, ch, length, info, m2)}


NODE_CLASS_MAPPINGS = {
    "H3MaskInpaint": H3MaskInpaint,
    "H3LatentPin": H3LatentPin,
    "H3MatchSource": H3MatchSource,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MaskInpaint": "H3 Mask Inpaint (region replace)",
    "H3LatentPin": "H3 Latent Pin (cuts — read description)",
    "H3MatchSource": "H3 Match Source Clip",
}
