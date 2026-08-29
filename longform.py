"""H3 workflow utilities: assemble, audio slice, take seeds, resolution.

Small nodes that close the gaps around the chain workflow. Each one exists because
a step was manual and easy to get wrong, or because a measured finding was
decaying into folklore.
"""

import logging
import re

import torch

from .avlatent import av
from .chunkplan import describe as describe_plan
from .chunkplan import find_cuts, plan as build_plan, tokens_per_frame
from .crop import H3_CANVAS_MP
from .geometry import canvas_for_megapixels
from .timing import (FPS, align_frames, av_aligned_runs_through, describe,
                     is_av_aligned, snap_av_aligned)

# measured on the chain recipe with everything else held constant
RESOLUTIONS = [
    "640x1120  (chain default — clean joins)",
    "512x896   (cheapest; 289s/8s clip)",
    "640x1120  portrait 4:7",
    "768x1152  (produced CUTS in chains)",
    "768x1344  (documented cap; no better)",
    "1344x768  landscape (every reference workflow)",
    "custom",
]


class H3Assemble:
    """Join rendered links end to end into one continuous take."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "link_1": ("IMAGE",),
            },
            "optional": {
                **{f"link_{i}": ("IMAGE",) for i in range(2, 9)},
                "audio": ("AUDIO", {"tooltip": "One track for the WHOLE take. Trimmed or "
                                               "padded to the assembled length."}),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 64,
                                "tooltip": "Frames to drop from the END of each link "
                                           "except the last. Leave 0 — the chain recipe "
                                           "has no overlap to trim, and trimming removes "
                                           "the frame the next link was keyed from."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Concatenate link renders into one clip, no overlap and no crossfade. "
                   "The chain recipe keyframes each link from the previous link's LAST "
                   "frame, so the links already butt together — blending them was "
                   "measurably worse (CJ on the blended assembly: 'dizzying').")

    def go(self, link_1, audio=None, trim_frames=0, **rest):
        links = [link_1]
        for i in range(2, 9):
            t = rest.get(f"link_{i}")
            if t is not None:
                links.append(t)
        w, h = links[0].shape[2], links[0].shape[1]
        for i, t in enumerate(links):
            if t.shape[1] != h or t.shape[2] != w:
                raise ValueError(
                    f"link_{i + 1} is {t.shape[2]}x{t.shape[1]}, link_1 is {w}x{h} — "
                    f"every link must render at the same resolution.")
        parts = []
        for i, t in enumerate(links):
            last = (i == len(links) - 1)
            parts.append(t if last or trim_frames <= 0 else t[:-trim_frames])
        video = torch.cat(parts, dim=0)
        n = video.shape[0]

        out_audio = audio
        if audio is not None:
            wav = audio["waveform"]
            sr = int(audio.get("sample_rate", 32000))
            need = int(round(n / FPS * sr))
            have = wav.shape[-1]
            if have < need:
                wav = torch.nn.functional.pad(wav, (0, need - have))
            out_audio = {"waveform": wav[..., :need], "sample_rate": sr}

        info = (f"{len(links)} link(s) -> {n} frames ({n / FPS:.2f}s) at {w}x{h}"
                + (f", trimmed {trim_frames}/link" if trim_frames else "")
                + (f", audio {out_audio['waveform'].shape[-1] / sr:.2f}s" if audio else
                   ", no audio"))
        return {"ui": {"h3char": [info]}, "result": (video, out_audio, info)}


class H3AudioSlice:
    """Work out which part of a soundtrack a given link needs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "link_index": ("INT", {"default": 0, "min": 0, "max": 63,
                               "tooltip": "Same value as on H3 Long-form Links."}),
                "seconds_per_link": ("FLOAT", {"default": 14.375, "min": 1.0, "max": 120.0,
                                     "step": 0.5,
                                     "tooltip": "14.375 s = 345 frames, the AV-aligned run "
                                                "nearest the 15 s we used to use."}),
                "av_aligned": ("BOOLEAN", {"default": True,
                               "tooltip": "Snap the link length to a run that lands exactly "
                                          "on BOTH the 24 fps video grid and the 40 Hz "
                                          "audio grid (39, 90, 141, 192, 243, 294, 345...). "
                                          "Off-grid lengths round the audio latent, so each "
                                          "link's audio is a fraction of a step out and the "
                                          "error accumulates down the chain."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional — only to check the track is long "
                                               "enough for this link."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("offset_seconds", "length", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("offset_seconds for H3 Audio Lock, from the link index. Doing this "
                   "arithmetic by hand silently gives the wrong link the wrong music. "
                   "The frame count is snapped to the 17n+5 grid — and, with av_aligned "
                   "on, to a run whose end also lands exactly on the 40 Hz audio grid, so "
                   "the offsets do not drift away from the picture over a long chain.")

    def go(self, link_index, seconds_per_link, av_aligned=True, audio=None):
        frames = align_frames(seconds_per_link)
        if av_aligned:
            frames = snap_av_aligned(frames)
        actual = frames / FPS                    # 14.375 asked -> 14.375 rendered
        offset = link_index * actual
        info = f"link {link_index}: offset {offset:.3f}s | " + describe(frames)
        if not is_av_aligned(frames):
            info += ("\nWARNING: this length is off the audio grid, so the offset above "
                     "drifts from the picture a little further on every link. Turn "
                     "av_aligned on, or pick from: "
                     + ", ".join(str(n) for n in av_aligned_runs_through(400)))
        if audio is not None:
            sr = int(audio.get("sample_rate", 32000))
            total = audio["waveform"].shape[-1] / sr
            end = offset + actual
            info += f" | track {total:.1f}s"
            if end > total + 0.01:
                info += (f" — TOO SHORT, link {link_index} needs audio out to "
                         f"{end:.1f}s. The lock will pad or repeat.")
        return {"ui": {"h3char": [info]}, "result": (float(offset), frames, info)}


class H3Take:
    """Seeds for re-rolling: one seed per take, held constant across a take's links."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_seed": ("INT", {"default": 2024, "min": 0,
                                      "max": 0xffffffffffffffff}),
                "take": ("INT", {"default": 1, "min": 1, "max": 999,
                         "tooltip": "Bump this to re-roll. The seed changes between takes "
                                    "but stays IDENTICAL across the links within a take, "
                                    "which is what continuity needs."}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Re-rolling is the normal workflow, not an exception: clipping through "
                   "props measured at roughly a 50% per-sample rate across four seeds at "
                   "fixed settings, and it did not track motion, resolution or anything "
                   "else we could change. So generate takes and keep the clean one — but "
                   "keep the seed constant WITHIN a take, because varying it per link "
                   "broke continuity.")

    def go(self, base_seed, take):
        # a plain hash rather than base+take: adjacent seeds are not independent
        # draws, and a re-roll wants a genuinely different sample
        seed = (base_seed * 6364136223846793005 + take * 1442695040888963407) \
            % 0xffffffffffffffff
        info = f"take {take}: seed {seed} (same on every link of this take)"
        return {"ui": {"h3char": [info]}, "result": (int(seed), info)}


class H3Resolution:
    """Canvas picker with what each option measured."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (RESOLUTIONS, {"default": RESOLUTIONS[0]}),
                "custom_width": ("INT", {"default": 640, "min": 32, "max": 4096,
                                         "step": 32}),
                "custom_height": ("INT", {"default": 1120, "min": 32, "max": 4096,
                                          "step": 32}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Pick a canvas, snapped to 32. Carries what each one measured so the "
                   "findings do not decay into folklore: 640x1120 chained cleanly, "
                   "768x1152 produced cuts in chains with everything else equal, and the "
                   "open weights are 768p-class — 768x1344 is the documented cap and was "
                   "no better than 768x1152.\n\nIf you size by AREA rather than by "
                   "picking from a list, use H3 Canvas instead: it takes megapixels and "
                   "an aspect ratio and reports what the 32-px grid actually delivered.")

    def go(self, preset, custom_width, custom_height):
        notes = ""
        if preset.startswith("custom"):
            w, h = custom_width, custom_height
        else:
            m = re.match(r"(\d+)x(\d+)", preset)
            w, h = int(m.group(1)), int(m.group(2))
            notes = preset[m.end():].strip()
        w = max(32, round(w / 32) * 32)
        h = max(32, round(h / 32) * 32)
        mp = w * h / 1e6
        info = f"{w}x{h}  {mp:.2f} MP  {notes}".strip()
        if min(w, h) > 768:
            info += " | WARNING: short edge over 768 — the open weights are 768p-class."
        if mp > 1.05:
            info += " | over the 768x1344 cap, which measured no better."
        return {"ui": {"h3char": [info]}, "result": (w, h, info)}


class H3Canvas:
    """Canvas from an AREA budget and a shape, instead of from a list.

    A preset list answers the wrong question. What costs money is AREA: H3
    denoises every token every step, a canvas contributes `(w/32)*(h/32)` tokens
    per latent frame, and attention goes as roughly the square of the sequence.
    The aspect ratio is nearly free by comparison. So the two decisions are
    separate and this node keeps them separate — set the budget, then set the
    shape.

    THE 32-PX GRID IS THE POINT
      Solving w*h = megapixels and w/h = aspect and then rounding each axis to 32
      independently moves both the area and the ratio. 0.62 MP at 9:16 delivers
      0.61; some requests land 5% out. That is not an error to hide, it is the
      number you are actually paying for, so `info` reports what was delivered
      rather than what was asked for.

    Capped at H3's own canvas ceiling (`MAX_PIXELS`, 768x1344). Past it the model
    is out of its trained range and the cost is quadratic, so the request is
    clamped and `info` says it was — a silent clamp would make the reported
    numbers describe a render nobody asked for.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {"default": 0.62, "min": 0.05,
                               "max": H3_CANVAS_MP, "step": 0.01,
                               "tooltip": "Area budget. This is the cost dial: "
                                          "0.62 MP is roughly 640x1120, and "
                                          "doubling it roughly quadruples the "
                                          "attention. Capped at H3's own "
                                          f"{H3_CANVAS_MP:.2f} MP canvas."}),
                "aspect_w": ("INT", {"default": 9, "min": 1, "max": 4096,
                             "tooltip": "Ratio, not pixels. 9:16 portrait, "
                                        "16:9 landscape, 1:1 square."}),
                "aspect_h": ("INT", {"default": 16, "min": 1, "max": 4096}),
            },
            "optional": {
                "like_image": ("IMAGE", {"tooltip": "Take the aspect ratio from "
                                         "this image instead of aspect_w / "
                                         "aspect_h, which are then IGNORED. The "
                                         "megapixels still come from the widget "
                                         "— this matches the shape of a source "
                                         "clip, not its size."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "tokens_per_frame", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("A render canvas from megapixels plus an aspect ratio, snapped to "
                   "32. Reports the area you actually got, the per-latent-frame token "
                   "count, and how far it sits from H3's canvas cap. Wire width/height "
                   "into the conditioning node and into H3 Chunk Plan so the size is "
                   "set in one place.")

    def go(self, megapixels, aspect_w, aspect_h, like_image=None):
        aw, ah = int(aspect_w), int(aspect_h)
        shape_from = f"{aw}:{ah}"
        if like_image is not None:
            aw, ah = int(like_image.shape[2]), int(like_image.shape[1])
            shape_from = f"{aw}:{ah} from like_image"

        asked = float(megapixels)
        w, h, used = canvas_for_megapixels(asked, aw, ah, multiple=32,
                                           cap_mp=H3_CANVAS_MP)
        cells = tokens_per_frame(w, h)
        got = w * h / 1e6

        info = (f"{w}x{h}  {got:.3f} MP  {cells:,} tokens per latent frame  "
                f"({shape_from} -> {w / h:.3f})")
        if used < asked - 1e-9:
            info += (f"\n  CLAMPED: {asked:.2f} MP asked, {used:.2f} MP used — that is "
                     f"H3's canvas cap (MAX_PIXELS 768x1344) and past it the model is "
                     f"out of its trained range.")
        # the delivered area is the number worth reading, so say how far the grid
        # moved it rather than leaving the reader to assume they got what they asked
        drift = got - used
        if abs(drift) >= 0.005:
            info += (f"\n  the 32-px grid moved {used:.2f} MP to {got:.3f} "
                     f"({drift * 100 / used:+.1f}%)")
        head = H3_CANVAS_MP - got
        if head >= 0:
            info += f"\n  {head:.3f} MP below the {H3_CANVAS_MP:.2f} MP canvas cap"
        else:
            # snapping to 32 rounds to NEAREST, so a request AT the cap can land a
            # hair over it (768x1344 is 1.032). Say which of the two it is.
            info += (f"\n  {-head:.3f} MP ABOVE the {H3_CANVAS_MP:.2f} MP canvas cap"
                     + (" — the 32-px rounding, not a request past it"
                        if used <= H3_CANVAS_MP else ""))
        return {"ui": {"h3char": [info]},
                "result": (int(w), int(h), int(cells), info)}


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
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Pin a real soundtrack into the AV latent so only video is "
                   "generated. Advance offset_seconds per clip to keep music "
                   "continuous across a chain.")

    def go(self, latent, audio_vae, audio, offset_seconds, strength):
        import comfy.nested_tensor
        video, aud = av(latent["samples"])
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
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = "Last frame of a clip, for keyframing the next one."

    def go(self, images, from_end):
        i = max(0, images.shape[0] - int(from_end))
        return (images[i:i + 1],)


class H3ChunkPlan:
    """Work out where to cut a long clip into chunks, and say so before you run.

    Pure arithmetic plus frame differencing — no models, no VAE, seconds to run.
    Its `info` is the deliverable: it tells you how many passes you are about to
    commit to, where each one lands, which are off the audio grid, and how strong
    your references will be in each. Reading that costs nothing; discovering it
    from a render that took an hour does not.

    WHY THE MODE MATTERS
      `fixed` keeps every chunk on both clocks but lets a chunk straddle a scene
      change, which means ONE prompt describing TWO scenes. `scene` ends chunks
      at cuts so the prompt always matches its content, at the cost of most
      chunks landing off the frame grid and padding up.

      In a V2V swap the source audio is PRESERVED rather than generated, and
      reassembly happens at exact sample boundaries, so an off-grid chunk costs a
      rounding at that chunk and nothing downstream. That is what makes `scene`
      affordable here when it would not be in a generative chain.

    Cut detection is mean absolute frame difference, thresholded. Hard cuts only
    — a dissolve spreads its change over many frames and will not trip it, which
    `info` says rather than leaving you to infer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # min 5 with step 17 IS the legal-run grid (17n+5), so the arrows
            # can only land on one. A typed value still snaps in plan(), which
            # says so in `info` -- the step just stops it happening silently.
            "chunk_frames": ("INT", {"default": 90, "min": 5, "max": 3600, "step": 17,
                             "tooltip": "Target chunk size, a legal run (17n+5). "
                                        "39/90/141/192/243/294/"
                                        "345 land on BOTH clocks. Shorter chunks "
                                        "also hold identity better — a reference "
                                        "is a fixed token count while the target "
                                        "grows, so its share falls as chunks get "
                                        "longer."}),
            "chunk_mode": (["scene", "fixed"], {"default": "scene",
                            "tooltip": "scene: chunks end at cuts, so a prompt "
                                       "never describes two scenes. fixed: uniform "
                                       "size, may straddle a cut."}),
        }, "optional": {
            "source_images": ("IMAGE", {"tooltip": "The clip to chunk. Leave "
                                        "UNWIRED for fresh generation and set "
                                        "total_frames instead — there is no "
                                        "source to slice, so chunks are just a "
                                        "schedule."}),
            "total_frames": ("INT", {"default": 0, "min": 0, "max": 36000,
                             "step": 1,
                             "tooltip": "Fresh generation only: how long the "
                                        "finished piece should be. Free-form — "
                                        "the CHUNKS have to be legal runs, the "
                                        "total does not (564 = 4x141 is not one "
                                        "itself). A piece shorter than one chunk "
                                        "is the exception: there the total IS the "
                                        "run, and it grows up to the next legal "
                                        "one. Ignored when source_images is "
                                        "wired."}),
            "scene_threshold": ("FLOAT", {"default": 0.12, "min": 0.005, "max": 1.0,
                                "step": 0.005,
                                "tooltip": "Mean absolute frame difference that "
                                           "counts as a cut, 0-1. Lower finds more. "
                                           "Check the detected list in `info` "
                                           "against the clip before trusting it."}),
            "min_chunk": ("INT", {"default": 39, "min": 5, "max": 3600, "step": 17,
                          "tooltip": "Also a legal run, because a shot this "
                                     "length becomes a chunk of its own. "
                                     "Shots shorter than this merge into a "
                                     "neighbour rather than becoming a chunk too "
                                     "short to be worth a pass."}),
            "render_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                             "tooltip": "Report only — token cost and reference "
                                        "share. IGNORED when source_images is "
                                        "wired, because the size is already "
                                        "known from the clip. Fresh generation "
                                        "has no clip, so wire H3 Canvas here and "
                                        "into the conditioning node so the size "
                                        "is set once. 0 = skip the columns."}),
            "render_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                              "tooltip": "See render_width."}),
            "ref_tokens": ("INT", {"default": 0, "min": 0, "max": 1000000,
                           "tooltip": "Reference token count from H3 Reference "
                                      "Budget, to report each chunk's ref share."}),
            # APPENDED, like everything after it. widgets_values is positional,
            # so a widget inserted anywhere else shifts every one after it in
            # every saved workflow -- silently, into fields that still look
            # plausible.
            "context": (["22", "39", "5", "1", "0"], {"default": "22",
                         "tooltip": "Frames each chunk carries from the previous "
                                    "one's finished output, to stop the seam "
                                    "restarting. Chunks OVERLAP by this much and "
                                    "the join drops it. Only 39/22/5/1 encode "
                                    "distinctly. 1 is a still keyframe: it fixes "
                                    "position but a keyframe-only chain eroded "
                                    "motion 19% down the links, where 22 held "
                                    "flat. 0 turns it off."}),
            "cut_frames": ("STRING", {"default": "", "multiline": False,
                           "tooltip": "Frame indices where a new SHOT begins, "
                                      "comma separated. `scene` mode finds these "
                                      "by differencing a source clip, which "
                                      "generating from nothing cannot do — this "
                                      "is how a T2V take gets real shot "
                                      "boundaries. A chunk that starts a shot "
                                      "carries pin 0, so the chain resets there: "
                                      "no inherited prefix, and the contrast "
                                      "carry measured across chunks stops with "
                                      "it. Ignored when a source clip supplies "
                                      "its own cuts."}),
        }}

    RETURN_TYPES = ("H3_CHUNK_PLAN", "INT", "STRING")
    RETURN_NAMES = ("plan", "chunk_count", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Plan where a long clip gets cut into chunks, aligned to scene "
                   "changes or to a fixed size. Reports the plan before you run it.")

    def go(self, chunk_frames, chunk_mode, source_images=None, total_frames=0,
           scene_threshold=0.12, min_chunk=39, render_width=0, render_height=0,
           ref_tokens=0, context="22", cut_frames=""):
        n = int(source_images.shape[0]) if source_images is not None else int(total_frames)
        if n <= 0:
            msg = ("Nothing to plan. Wire source_images for a V2V pass, or set "
                   "total_frames for a fresh generation.")
            return {"ui": {"h3char": [msg]},
                    "result": ({"chunks": [], "info": {}, "total_frames": 0}, 0, msg)}

        cuts = []
        if cut_frames.strip():
            try:
                cuts = sorted({int(x) for x in cut_frames.replace(";", ",").split(",")
                               if x.strip()})
            except ValueError:
                raise ValueError(
                    f"H3 Chunk Plan: `cut_frames` must be whole frame numbers "
                    f"separated by commas — got {cut_frames!r}.")
            cuts = [c for c in cuts if 0 < c < n]
        if chunk_mode == "scene" and not cuts and source_images is not None and n > 1:
            # mean |delta| per frame against its predecessor. Cheap, and a hard
            # cut changes the whole picture in one frame so it stands well clear
            # of ordinary motion.
            a = source_images[1:, ..., :3].float()
            b = source_images[:-1, ..., :3].float()
            deltas = (a - b).abs().mean(dim=(1, 2, 3)).tolist()
            cuts = find_cuts([0.0] + deltas, float(scene_threshold))

        # Growing the tail forward is only honest with nothing to cut: fresh
        # generation invents the frames, a V2V pass would be asking for footage
        # past the end of the clip.
        #
        # No source clip also means no source AUDIO, so the audio is generated
        # and every off-grid link's rounding accumulates down the chain — which
        # a V2V pass does not suffer, because it slices the source track at exact
        # sample boundaries. Same trigger, different reason: kept as two flags so
        # the reason stays visible.
        fresh = source_images is None
        chunks, info = build_plan(n, int(chunk_frames), chunk_mode, cuts=cuts,
                                  min_chunk=int(min_chunk), context=int(context),
                                  grow_tail=fresh, generated_audio=fresh)
        # A V2V pass already knows its render size — it is the size of the clip
        # in hand, [T, H, W, C] — so asking for it again is a second copy that
        # can only be stale. Nobody filled the widgets in, so the token and
        # reference-share columns never appeared on the workflow that most needed
        # them. The widgets stay for fresh generation, where there is no clip;
        # they are INT so H3 Canvas can drive them.
        if source_images is not None:
            render = (int(source_images.shape[2]), int(source_images.shape[1]))
            render_from = "from source_images"
        elif render_width and render_height:
            render = (int(render_width), int(render_height))
            render_from = "from the widgets"
        else:
            render, render_from = None, ""
        text = describe_plan(chunks, info, render=render, ref_tokens=int(ref_tokens),
                             render_from=render_from)

        generated = sum(c["run"] for c in chunks)
        if generated > n:
            text += (f"\n  {generated} frames generated for {n} of content "
                     f"({100 * (generated / float(n) - 1):.1f}% padding)")

        logging.info("H3ChunkPlan: %s", text.splitlines()[0])
        payload = {"chunks": chunks, "info": info, "total_frames": n}
        return {"ui": {"h3char": [text]},
                "result": (payload, len(chunks), text)}


class H3SeamCheck:
    """Put the chunk seams in front of your eye, and say how big they are.

    WHY THIS EXISTS
      A join in a 300-frame clip is one frame out of 300, and finding it by
      scrubbing is guesswork -- you end up unsure whether what you saw was a seam
      or the motion. The plan already knows exactly which frames are joins, so
      this cuts a strip either side of each one and reports the frame-to-frame
      difference there against the clip's own distribution.

    WHAT THE NUMBERS DO AND DO NOT DO
      Reported as DATA. Every quality proxy this project has built has failed, so
      there is no threshold here and no verdict -- a delta is compared to the
      SAME CLIP's median and p95 and that is all. A join inside the clip's own
      range of motion is not evidence of a good seam, only that the difference is
      not unusual for this footage. The strip is what you actually judge.

    THE ONE THING THE DELTA COLUMN DECIDES
      Two different faults land at two different frames, and telling them apart
      is the point of the `+1 +2 +3` columns:

      A spike AT the join is a chunk discontinuity -- the two chunks disagree
      about what is in frame, and the pin is not doing its job.

      A spike ONE TO THREE FRAMES AFTER it is the pin RELEASING. Inside a chunk
      the first `pin` frames are held at denoise mask 0 and the rest are
      generated, which is a hard edge in time; H3LatentPin in this pack carries a
      warning from nine rounds of measurement that the model reproduces a pinned
      run and then splices to its own version at the frame the pin ends. The join
      currently cuts at exactly that frame, so the release transient becomes the
      first frame you see. That is a planner fix, not a prompt or setting one,
      and these columns are how you tell it apart from the other case.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The JOINED clip — H3 Chunk Close's "
                                            "output, or after the uncrop."}),
        }, "optional": {
            "plan": ("H3_CHUNK_PLAN", {"tooltip": "From H3 Chunk Plan, to locate "
                     "the joins. Without it this still reports the delta "
                     "distribution, but it cannot tell you where to look."}),
            "frames_each_side": ("INT", {"default": 4, "min": 1, "max": 16,
                                 "tooltip": "How many frames either side of each "
                                            "join go in the strip."}),
            "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0,
                      "step": 0.05,
                      "tooltip": "Shrink the contact sheet. A 900px-wide strip of "
                                 "eight frames is unreadable at 1.0."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("strips", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("One labelled strip per chunk join, plus the frame-to-frame "
                   "difference at each one against the clip's own spread.")

    @staticmethod
    def _deltas(images):
        """Mean |frame - previous frame|, one value per frame after the first."""
        a = images[1:, ..., :3].float()
        b = images[:-1, ..., :3].float()
        return (a - b).abs().mean(dim=(1, 2, 3))

    @staticmethod
    def _joins(plan, n):
        """Where each chunk's kept frames begin IN THE JOINED CLIP.

        Not the same as `keep_from`, which is a SOURCE index. They coincide while
        coverage starts at 0 and stays contiguous, but the join is the cumulative
        length of what came before it, so compute that and stay correct if the
        plan ever changes.
        """
        chunks = (plan or {}).get("chunks") or []
        joins, pos = [], 0
        for i in range(len(chunks) - 1):
            pos += int(chunks[i]["end"]) - int(chunks[i]["keep_from"])
            if 0 < pos < n:
                joins.append((pos, chunks[i + 1]))
        return joins

    def go(self, images, plan=None, frames_each_side=4, scale=0.5):
        import torch.nn.functional as Fn

        n = int(images.shape[0])
        if n < 2:
            msg = "H3 SEAM CHECK: needs at least two frames."
            return {"ui": {"h3char": [msg]}, "result": (images[:1], msg)}

        d = self._deltas(images)
        srt = d.sort().values
        med = float(srt[len(srt) // 2])
        p95 = float(srt[min(len(srt) - 1, int(len(srt) * 0.95))])
        joins = self._joins(plan, n)

        L = [f"H3 SEAM CHECK — {n} frames joined, {len(joins)} seam(s)"]
        L.append(f"  frame-to-frame delta over the whole clip: "
                 f"median {med:.4f}  p95 {p95:.4f}  max {float(srt[-1]):.4f}")
        if not joins:
            L.append("  no plan wired (or a single chunk) — nothing to point at. "
                     "Wire H3 Chunk Plan's `plan` to locate the joins.")

        k = max(1, int(frames_each_side))
        rows = []
        for i, (j, chunk) in enumerate(joins, 1):
            # d[x] is the step from frame x to x+1, so the join's own step is d[j-1]
            at = float(d[j - 1])
            after = [float(d[min(len(d) - 1, j - 1 + t)]) for t in (1, 2, 3)]
            pin = int(chunk.get("pin", 0))
            L.append(f"  seam {i}  frame {j:>5}   at join {at:.4f} "
                     f"({at / p95 if p95 else 0:.2f}x p95)"
                     f"   +1 {after[0]:.4f}  +2 {after[1]:.4f}  +3 {after[2]:.4f}"
                     + (f"   [{pin}f pinned]" if pin else ""))

            a, b = max(0, j - k), min(n, j + k)
            strip = images[a:b, ..., :3]
            row = torch.cat([strip[t] for t in range(strip.shape[0])], dim=1)
            # a bright column exactly on the cut, so the strip is readable at a
            # glance rather than counted through
            w = int(images.shape[2])
            x = (j - a) * w
            if 0 < x < row.shape[1]:
                row[:, max(0, x - 1):x + 1, :] = torch.tensor(
                    [1.0, 0.2, 0.2], device=row.device, dtype=row.dtype)
            rows.append(row)

        if not rows:
            blank = images[:1]
            return {"ui": {"h3char": ["\n".join(L)]},
                    "result": (blank, "\n".join(L))}

        width = max(r.shape[1] for r in rows)
        padded = [Fn.pad(r.permute(2, 0, 1), (0, width - r.shape[1])).permute(1, 2, 0)
                  for r in rows]
        sheet = torch.cat(padded, dim=0).unsqueeze(0)
        s = float(scale)
        if s < 0.999:
            sheet = Fn.interpolate(sheet.permute(0, 3, 1, 2),
                                   scale_factor=s, mode="bilinear",
                                   align_corners=False).permute(0, 2, 3, 1)

        L.append("")
        L.append("  a spike AT the join is the two chunks disagreeing; a spike at")
        L.append("  +1..+3 is the pin releasing, which is a different fault. The")
        L.append("  red line in each strip is the cut.")
        text = "\n".join(L)
        logging.info("H3SeamCheck: %s", L[0])
        return {"ui": {"h3char": [text]}, "result": (sheet, text)}



NODE_CLASS_MAPPINGS = {"H3Assemble": H3Assemble, "H3AudioSlice": H3AudioSlice,
                       "H3Take": H3Take, "H3Resolution": H3Resolution,
                       "H3Canvas": H3Canvas,
                       "H3AudioLock": H3AudioLock,
                       "H3ChainFrame": H3ChainFrame,
                       "H3ChunkPlan": H3ChunkPlan,
                       "H3SeamCheck": H3SeamCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Assemble": "H3 Assemble Links",
                              "H3AudioSlice": "H3 Audio Slice (per link)",
                              "H3Take": "H3 Take (re-roll seed)",
                              "H3Resolution": "H3 Resolution",
                              "H3Canvas": "H3 Canvas (megapixels)",
                              "H3AudioLock": "H3 Audio Lock",
                              "H3ChainFrame": "H3 Chain Frame",
                              "H3ChunkPlan": "H3 Chunk Plan",
                              "H3SeamCheck": "H3 Seam Check"}
