"""H3 workflow utilities: assemble, audio slice, take seeds, resolution.

Small nodes that close the gaps around the chain workflow. Each one exists because
a step was manual and easy to get wrong, or because a measured finding was
decaying into folklore.
"""

import re

import torch

from .geometry import crop_to_multiple
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

    Dimensions are CROPPED to a multiple of 32, never resized. In an inpaint most
    of the output IS the source — everything outside the mask is pinned from these
    exact pixels — so resampling would soften the whole frame to accommodate a
    region that gets regenerated anyway. Cropping leaves the retained pixels
    untouched, and costs a sliver of frame edge instead. (An earlier version of
    this node refused non-/32 input outright, which just moved the problem.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The source clip."}),
        }, "optional": {
            "av_aligned": ("BOOLEAN", {"default": False,
                           "tooltip": "Also trim to a run that lands exactly on the 40 Hz "
                                      "audio grid. Costs up to 51 more frames, so it is "
                                      "off by default — worth it when this clip is a link "
                                      "in a chain, not when you are masking a one-off."}),
            "mask": ("MASK", {"tooltip": "Optional — cropped and trimmed identically, so "
                                         "it stays pixel-aligned with the frames."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "mask", "width", "height", "length", "info")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/long-form"
    DESCRIPTION = ("Read a source clip's width, height and a legal frame count, cropping "
                   "to /32 and trimming to the frame grid. Wire all three into the H3 "
                   "conditioning node so an inpaint can never be shape-mismatched.")

    def go(self, images, av_aligned=False, mask=None):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        notes = []

        try:
            x0, y0, cw, ch = crop_to_multiple(w, h, 32)
        except ValueError:
            raise ValueError(f"source is {w}x{h}; H3 needs at least 32x32 after cropping "
                             f"to a multiple of 32.")
        if (cw, ch) != (w, h):
            images = images[:, y0:y0 + ch, x0:x0 + cw, :]
            if mask is not None:
                mask = mask[..., y0:y0 + ch, x0:x0 + cw]
            notes.append(f"centre-cropped {w}x{h} -> {cw}x{ch} (no resample)")

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
        if mask is not None:
            mask = mask[:length]

        info = f"{cw}x{ch}, " + describe(length)
        if notes:
            info += " — " + "; ".join(notes)
        if mask is None:
            mask = torch.zeros(length, ch, cw)
        return {"ui": {"h3char": [info]},
                "result": (images, mask, cw, ch, length, info)}


NODE_CLASS_MAPPINGS = {"H3MatchSource": H3MatchSource, "H3Assemble": H3Assemble, "H3AudioSlice": H3AudioSlice,
                       "H3Take": H3Take, "H3Resolution": H3Resolution}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MatchSource": "H3 Match Source Clip",
                              "H3Assemble": "H3 Assemble Links",
                              "H3AudioSlice": "H3 Audio Slice (per link)",
                              "H3Take": "H3 Take (re-roll seed)",
                              "H3Resolution": "H3 Resolution"}
