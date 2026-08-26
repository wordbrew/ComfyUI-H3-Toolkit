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
from .chunkplan import find_cuts, plan as build_plan
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
                   "no better than 768x1152.")

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
            "source_images": ("IMAGE", {"tooltip": "The whole clip."}),
            "chunk_frames": ("INT", {"default": 90, "min": 5, "max": 3600, "step": 1,
                             "tooltip": "Target chunk size. 39/90/141/192/243/294/"
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
            "scene_threshold": ("FLOAT", {"default": 0.12, "min": 0.005, "max": 1.0,
                                "step": 0.005,
                                "tooltip": "Mean absolute frame difference that "
                                           "counts as a cut, 0-1. Lower finds more. "
                                           "Check the detected list in `info` "
                                           "against the clip before trusting it."}),
            "min_chunk": ("INT", {"default": 39, "min": 5, "max": 3600,
                          "tooltip": "Shots shorter than this merge into a "
                                     "neighbour rather than becoming a chunk too "
                                     "short to be worth a pass."}),
            "render_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                             "tooltip": "Only to report reference share. 0 = skip."}),
            "render_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "ref_tokens": ("INT", {"default": 0, "min": 0, "max": 1000000,
                           "tooltip": "Reference token count from H3 Reference "
                                      "Budget, to report each chunk's ref share."}),
        }}

    RETURN_TYPES = ("H3_CHUNK_PLAN", "INT", "STRING")
    RETURN_NAMES = ("plan", "chunk_count", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Plan where a long clip gets cut into chunks, aligned to scene "
                   "changes or to a fixed size. Reports the plan before you run it.")

    def go(self, source_images, chunk_frames, chunk_mode, scene_threshold=0.12,
           min_chunk=39, render_width=0, render_height=0, ref_tokens=0):
        n = int(source_images.shape[0])

        cuts = []
        if chunk_mode == "scene" and n > 1:
            # mean |delta| per frame against its predecessor. Cheap, and a hard
            # cut changes the whole picture in one frame so it stands well clear
            # of ordinary motion.
            a = source_images[1:, ..., :3].float()
            b = source_images[:-1, ..., :3].float()
            deltas = (a - b).abs().mean(dim=(1, 2, 3)).tolist()
            cuts = find_cuts([0.0] + deltas, float(scene_threshold))

        chunks, info = build_plan(n, int(chunk_frames), chunk_mode, cuts=cuts,
                                  min_chunk=int(min_chunk))
        render = (int(render_width), int(render_height)) if render_width and render_height else None
        text = describe_plan(chunks, info, render=render, ref_tokens=int(ref_tokens))

        generated = sum(c["run"] for c in chunks)
        if generated > n:
            text += (f"\n  {generated} frames generated for {n} of content "
                     f"({100 * (generated / float(n) - 1):.1f}% padding)")

        logging.info("H3ChunkPlan: %s", text.splitlines()[0])
        payload = {"chunks": chunks, "info": info, "total_frames": n}
        return {"ui": {"h3char": [text]},
                "result": (payload, len(chunks), text)}


NODE_CLASS_MAPPINGS = {"H3Assemble": H3Assemble, "H3AudioSlice": H3AudioSlice,
                       "H3Take": H3Take, "H3Resolution": H3Resolution,
                       "H3AudioLock": H3AudioLock,
                       "H3ChainFrame": H3ChainFrame,
                       "H3ChunkPlan": H3ChunkPlan}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Assemble": "H3 Assemble Links",
                              "H3AudioSlice": "H3 Audio Slice (per link)",
                              "H3Take": "H3 Take (re-roll seed)",
                              "H3Resolution": "H3 Resolution",
                              "H3AudioLock": "H3 Audio Lock",
                              "H3ChainFrame": "H3 Chain Frame",
                              "H3ChunkPlan": "H3 Chunk Plan"}
