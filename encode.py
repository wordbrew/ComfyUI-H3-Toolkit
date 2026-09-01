"""Images into an H3 AV latent — the front door img2img never had.

WHY THIS EXISTS
  Every H3 conditioning node hands you an EMPTY latent, because they are built
  for generating. Anything that starts from footage instead — a refine, an
  upscale, a second pass over a finished render — needs the opposite: the
  clip encoded INTO a latent the sampler can partially denoise.

  Until now the only route was `H3MaskInpaint`, which does it as a side effect
  of inpainting. That works, and it is what the first PDD upscale graph used,
  but it means passing a solid white mask to an inpainting node so it pins
  nothing, and taking a shape latent from a conditioning node that allocated it
  for a render that never happens. Two nodes and a dummy mask to say "encode
  this". Anyone reading the graph reasonably asks what the mask is for.

RESIZING LIVES HERE TOO, ON PURPOSE
  The latent's shape IS the render size — `[B, 24, T, h/16, w/16]` — so an
  upstream resize and this node cannot disagree without the result being
  wrong. Owning both means there is one number, and it comes out of the
  `width`/`height` outputs for the conditioning node to follow. A separate
  resize node would reintroduce exactly the mismatch this removes.

  Scaling ABOVE H3's 1.03 MP canvas is allowed and is the point on a refine
  pass: below denoise ~0.5 the structure is already in the latent and only RoPE
  positions extrapolate, which degrades softly. Generating from noise up there
  is what duplicates features. See crop.py's MAX_UPSCALE_MP.

AUDIO IS PINNED BY DEFAULT
  A video pass has no business resampling the dialogue. With `pin_audio` on and
  a soundtrack wired, the audio half of the latent is the encoded source and
  its denoise mask is zero, so sampling leaves it alone. That is the one piece
  of masking this node does, and it is doing real work.
"""

import logging

import torch
import torch.nn.functional as Fn

from .geometry import cover_crop
from .timing import audio_t as audio_len

CATEGORY = "MiniMax H3/video"


def fit_size(sw, sh, megapixels, grid):
    """Target w, h for a source of sw x sh, on the grid, at ~megapixels area.

    Aspect is preserved and both axes land on `grid`, which H3 requires (32).
    `megapixels` of 0 keeps the source's own size, still snapped.
    """
    grid = max(8, int(grid))

    def snap(v):
        return max(grid, int(round(v / grid)) * grid)

    if megapixels and megapixels > 0:
        scale = (float(megapixels) * 1e6 / max(1.0, sw * sh)) ** 0.5
        return snap(sw * scale), snap(sh * scale)
    return snap(sw), snap(sh)


class H3EncodeAV:
    """Encode a clip into an H3 video+audio latent, resized to the render size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The clip to encode. Frame COUNT is "
                                            "not resamplable — it sits on the "
                                            "VAE's 17n+5 grid — so a source that "
                                            "is off-grid is trimmed DOWN to the "
                                            "nearest legal run and the info "
                                            "output says by how much."}),
            "vae": ("VAE", {"tooltip": "The VIDEO vae."}),
            "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0,
                           "step": 0.05,
                           "tooltip": "0 = keep the source's size (still snapped "
                                      "to the grid). Above 0, the clip is scaled "
                                      "to about this area — that is the upscale. "
                                      "H3 GENERATES at 1.03 MP (768x1344); going "
                                      "past it is fine for a refine at low "
                                      "denoise and wrong for generating from "
                                      "noise. Attention cost is QUADRATIC in "
                                      "area, so 2.87 MP is ~8x the canvas."}),
            "divisible_by": ("INT", {"default": 32, "min": 8, "max": 128,
                             "step": 8, "tooltip": "H3 needs 32. Leave it."}),
        }, "optional": {
            "audio_vae": ("VAE", {"tooltip": "Needed to carry the source audio. "
                                             "Without it the audio half is "
                                             "silence and gets generated."}),
            "source_audio": ("AUDIO", {"tooltip": "The clip's own soundtrack."}),
            "pin_audio": ("BOOLEAN", {"default": True,
                          "tooltip": "Hold the source audio through sampling so "
                                     "a video pass cannot resample your "
                                     "dialogue. Needs audio_vae and "
                                     "source_audio wired. Turn OFF only if you "
                                     "want the soundtrack regenerated."}),
            "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                      "tooltip": "0 = derive from megapixels. Set both width "
                                 "and height to pin an exact size; they are "
                                 "still snapped to the grid, and the clip is "
                                 "centre-cropped to their aspect rather than "
                                 "stretched."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "length", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Encode a clip into an H3 video+audio latent for a refine or "
                   "upscale pass, resized to the render size. Wire width/height/"
                   "length into the H3 conditioning node so they cannot drift.")

    def go(self, images, vae, megapixels, divisible_by, audio_vae=None,
           source_audio=None, pin_audio=True, width=0, height=0):
        n = int(images.shape[0])
        # Trim DOWN, never up: the VAE takes 17n+5 and there is nothing to
        # invent at the tail. Reported rather than silent — losing frames off
        # the end of a take is the kind of thing that gets blamed on the model.
        run = n
        while run % 17 != 5 and run > 5:
            run -= 1
        if run < 5:
            raise ValueError(
                f"H3 Encode AV: {n} frame(s) is below the VAE's minimum run of 5.")
        dropped = n - run
        src = images[:run]

        sh, sw = int(src.shape[1]), int(src.shape[2])
        if width > 0 and height > 0:
            g = max(8, int(divisible_by))
            tw = max(g, int(round(width / g)) * g)
            th = max(g, int(round(height / g)) * g)
        else:
            tw, th = fit_size(sw, sh, megapixels, divisible_by)

        # Conform without stretching: centre-crop to the target ASPECT first,
        # which resamples nothing, then rescale if the size still differs.
        if (sw, sh) != (tw, th):
            x0, y0, cw, ch = cover_crop(sw, sh, tw, th)
            if (cw, ch) != (sw, sh):
                src = src[:, y0:y0 + ch, x0:x0 + cw, :]
                logging.info("H3EncodeAV: centre-cropped %dx%d -> %dx%d to match "
                             "the target aspect", sw, sh, cw, ch)
            if (cw, ch) != (tw, th):
                src = Fn.interpolate(src.movedim(-1, 1), size=(th, tw),
                                     mode="bicubic", align_corners=False,
                                     antialias=True).clamp(0, 1).movedim(1, -1)

        z = vae.encode(src)                                   # [1, 24, T, h, w]
        at = audio_len(run)

        # The audio half always exists — the model expects a pair — and is
        # silence unless a soundtrack is wired.
        za = torch.zeros([z.shape[0], 32, 2, at], device=z.device, dtype=z.dtype)
        mask_a = torch.ones_like(za)
        pinned = False
        if pin_audio and source_audio is not None and audio_vae is not None:
            wav = source_audio["waveform"]
            sr = int(source_audio["sample_rate"])
            vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
            if sr != vae_sr:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, vae_sr)
            enc = audio_vae.encode(wav[:1].movedim(1, -1))
            enc = enc[..., :at]
            if enc.shape[-1] < at:
                pad = at - enc.shape[-1]
                enc = torch.cat(
                    [enc, enc[..., -1:].expand(*enc.shape[:-1], pad)], dim=-1)
            za = enc.to(z.device, z.dtype)
            mask_a = torch.zeros_like(za)
            pinned = True

        # imported here rather than at the top of the method so an illegal
        # frame count reports ITSELF, instead of failing on a heavy import first
        import comfy.nested_tensor

        out = {"samples": comfy.nested_tensor.NestedTensor((z, za)),
               "noise_mask": comfy.nested_tensor.NestedTensor(
                   (torch.ones_like(z), mask_a))}

        mp = (tw * th) / 1e6
        info = (f"{tw}x{th} = {mp:.2f} MP, {run} frames -> {z.shape[2]} latent "
                f"frames, audio {at} steps\n"
                f"  audio: {'PINNED from source' if pinned else 'silence, will be generated'}")
        if (sw, sh) != (tw, th):
            info += f"\n  resized from {sw}x{th if sh == th else sh}"
        if dropped:
            info += (f"\n  TRIMMED {dropped} frame(s) off the end: {n} is not "
                     f"17n+5, nearest legal run below is {run}")
        if mp > 1.03:
            info += (f"\n  {mp / 1.03:.2f}x above H3's 1.03 MP canvas — intended "
                     f"for a REFINE at low denoise; generating from noise up "
                     f"here duplicates features. Cost is quadratic in area.")
        return {"ui": {"h3char": [info]},
                "result": (out, int(tw), int(th), int(run), info)}


NODE_CLASS_MAPPINGS = {"H3EncodeAV": H3EncodeAV}
NODE_DISPLAY_NAME_MAPPINGS = {"H3EncodeAV": "H3 Encode AV (images to latent)"}
