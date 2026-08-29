"""Measure what a chunk chain does to the picture over its own length.

One node so far, H3DriftCheck. It exists because a fault that develops ACROSS a
chain is invisible frame to frame -- nothing is wrong in any one frame, and the
first and last frames are 30 seconds apart so they are never on screen together.

Kept apart from longform.py deliberately: that module holds workflow plumbing,
this one holds measurement, and measurement code carries a different obligation
-- every number in here has to be traceable to how it was computed, because the
standing result in this project is that quality proxies fail.
"""

import logging

import torch

CATEGORY = "MiniMax H3/long-form"

REGIONS = ["torso band", "whole frame"]

# Fractions of the frame that are torso on a portrait character shot. Measured
# 2026-08-29: this band excludes sky, sand and floor, which is the whole reason
# it is here -- a bare R>G>B skin test put 50% of a warm beach frame in the mask,
# so "skin contrast" was really "sand contrast" and the number meant nothing.
BAND_TOP, BAND_BOTTOM, BAND_LEFT, BAND_RIGHT = 0.30, 0.72, 0.15, 0.85


def segments(plan, n):
    """Each chunk's span IN THE JOINED CLIP, as (start, stop, chunk).

    `start` and `end` in a plan are SOURCE indices and OVERLAP each other by the
    context, so indexing a joined clip with them lands in the wrong chunk from
    the second one onward. What actually reaches the join is `end - keep_from`
    per chunk, laid end to end -- the same arithmetic `H3SeamCheck._joins` uses
    to find the cuts, and the reason that method computes a running sum instead
    of trusting `keep_from`.

    The joined clip can also be SHORTER than the kept lengths sum to. With
    `grow_tail` the last chunk overshoots the ask -- 720 frames planned comes
    back as 753 kept -- and H3 Chunk Close trims to the ask, so clamp to `n` and
    drop whatever falls off the end rather than indexing past it.
    """
    out, pos = [], 0
    for chunk in (plan or {}).get("chunks") or []:
        kept = int(chunk["end"]) - int(chunk["keep_from"])
        start, stop = pos, min(n, pos + kept)
        pos += kept
        if start >= n:
            break
        if stop > start:
            out.append((start, stop, chunk))
    return out


def _region(images, region):
    """Crop to the region the numbers are allowed to come from."""
    x = images[..., :3]
    if region == "whole frame":
        return x
    h, w = int(x.shape[1]), int(x.shape[2])
    return x[:, int(h * BAND_TOP):int(h * BAND_BOTTOM),
             int(w * BAND_LEFT):int(w * BAND_RIGHT), :]


def _shrink(x, scale):
    import torch.nn.functional as Fn

    if scale >= 0.999:
        return x
    return Fn.interpolate(x.permute(0, 3, 1, 2), scale_factor=float(scale),
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def _detail(g):
    """|luma - 3x3 box blur|: micro-texture, with the exposure divided out.

    This is the quantity that works, after three that did not. Measured
    2026-08-29 across a 720-frame 4-chunk render whose skin the eye reads as
    "darker, more contrasted, gradual": mean luma is FLAT to slightly UP (+3.7%),
    so it is not exposure; saturation is FLAT to slightly DOWN (-3.3%), so it is
    not colour. Local contrast rose 25.6% over the same four chunks. The
    highlights hold while the shadow definition grows, which is exactly the shape
    that moves high-frequency energy and leaves the mean where it was.

    Built with `cat` rather than `F.pad(mode="replicate")` because pad wants a 4D
    batch and a [T,H,W] round trip is not worth it for a one-pixel border.
    Replicating the edge matters though: zero padding makes the frame border read
    as enormous contrast, which is the same trap `count_include_pad=False` exists
    for elsewhere in this pack.
    """
    p = torch.cat([g[:, :1, :], g, g[:, -1:, :]], dim=1)
    p = torch.cat([p[:, :, :1], p, p[:, :, -1:]], dim=2)
    h, w = int(g.shape[1]), int(g.shape[2])
    blur = p[:, 0:h, 0:w]
    for i in range(3):
        for j in range(3):
            if i or j:
                blur = blur + p[:, i:i + h, j:j + w]
    return torch.abs(g - blur / 9.0)


def _skin(band):
    """A tight skin mask, and the saturation it was cut from.

    Every bound here is doing work. R>G>B alone is warm-anything -- sand, wood,
    late sun -- so saturation is fenced from both sides (bare skin sits well
    inside 0.18-0.62 while sand sits below it and clothing dye above), and the
    max channel is fenced to drop crushed shadow and blown highlight, neither of
    which carries texture to measure.
    """
    mx = torch.amax(band, -1)
    mn = torch.amin(band, -1)
    sat = (mx - mn) / (mx + 1e-6)
    skin = ((band[..., 0] > band[..., 1]) & (band[..., 1] > band[..., 2])
            & (sat > 0.18) & (sat < 0.62) & (mx > 0.25) & (mx < 0.98))
    return skin, sat


def _floor(frames):
    """Fewest skin pixels a chunk may report a mean from.

    Scaled by length, because the failure this guards is a confident-looking
    figure averaged over a few stray warm pixels in a shot with no skin in it --
    and 30 pixels found across 200 frames is that, while 30 in one frame is
    merely a small subject. 100 per frame is roughly 0.2% of the band.
    """
    return max(1000, 100 * int(frames))


def _pct(value, reference):
    """A ratio column, or a dash where there is nothing to divide by.

    Zero is reachable: a flat region -- a solid colour card, a heavily denoised
    render, or a `scale` low enough to alias the texture away -- has no local
    contrast at all, and a percentage against it is undefined rather than
    infinite.
    """
    if reference is None or reference == 0.0:
        return "—"
    return f"{(value / reference - 1) * 100:+.1f}%"


def _measure(band):
    """(detail, luma, saturation, skin pixel count) over one segment's skin."""
    g = band.mean(-1)
    det = _detail(g)
    skin, sat = _skin(band)
    count = int(skin.sum())
    if count < _floor(int(band.shape[0])):
        return None, None, None, count
    return (float(det[skin].mean()), float(g[skin].mean()),
            float(sat[skin].mean()), count)


class H3DriftCheck:
    """Say whether the picture is drifting along the chain, and by how much.

    WHY THIS EXISTS
      A chunked take accumulates local contrast on skin, one step per chunk.
      Measured 2026-08-29 on a 720-frame 4-chunk render: 0.01239, 0.01392,
      0.01489, 0.01556 -- +12.4%, +20.2%, +25.6% against chunk 1, with the step
      shrinking (+12.4%, +6.9%, +4.5%) the way a repeated round trip does. The
      eye reads it as skin going darker and more contrasted, gradually, and it is
      invisible frame to frame: nothing is wrong in any single frame and the two
      frames that differ are 30 seconds apart.

    WHAT WAS TRIED AND FAILED
      Mean luma: FLAT to +3.7% -- the effect is not exposure. Saturation: FLAT to
      -3.3% -- not colour. A naive R>G>B skin mask: 50% of a warm beach frame
      classified as skin, so the figure tracked sand. All three are still
      reported here, as columns, precisely because they are the ones that do NOT
      move; a run where luma is what moved is a different fault.

    WHY PER CHUNK AND NOT PER SECOND
      Aggregated in 2-second blocks the same data looks like a continuous ramp,
      which points at the sampler or the prompt and led to a wrong diagnosis.
      Per chunk it is a staircase with the increment landing at each seam, and
      that is what identified it as chain accumulation -- something the join
      does, not something the render does.

    WHAT THESE NUMBERS DO NOT DO
      Report only. Every quality proxy this project has built has failed, so
      there is no threshold here, no verdict and no advice about what a good
      figure is. The numbers LOCALISE -- which chunk, how much, on which of the
      three quantities -- and the contact sheet is what you actually judge.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The JOINED clip — H3 Chunk Close's "
                                            "output, or after the uncrop."}),
        }, "optional": {
            "plan": ("H3_CHUNK_PLAN", {"tooltip": "From H3 Chunk Plan, to locate "
                     "the chunk boundaries. Without it the whole clip is one "
                     "segment and a number cannot be attributed to a chunk."}),
            "region": (REGIONS, {"default": "torso band",
                       "tooltip": "Where the numbers may come from. The torso "
                                  "band drops sky, sand and floor, which a skin "
                                  "test otherwise mistakes for skin."}),
            "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0,
                      "step": 0.05,
                      "tooltip": "Shrink before analysing, for speed. The "
                                 "absolute detail figure moves with this, so "
                                 "compare only within one run."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Per-chunk local contrast, luma and saturation on skin, plus "
                   "one frame per chunk side by side to check the claim against.")

    def go(self, images, plan=None, region="torso band", scale=0.5):
        n = int(images.shape[0])
        if n < 1:
            msg = "H3 DRIFT CHECK: no frames."
            return {"ui": {"h3char": [msg]}, "result": (images, msg)}

        segs = segments(plan, n)
        attributed = bool(segs)
        if not segs:
            segs = [(0, n, None)]

        rows, frames = [], []
        for i, (a, b, _chunk) in enumerate(segs, 1):
            # crop first, then shrink: the band is under a third of the area, and
            # resampling the full clip would briefly double an already large
            # tensor for pixels that get thrown away
            band = _shrink(_region(images[a:b], region), float(scale))
            rows.append((i, a, b) + _measure(band))
            frames.append(band[int(band.shape[0]) // 2])

        base = next((r[3] for r in rows if r[3] is not None), None)
        prev = None

        L = [f"H3 DRIFT CHECK — {n} frames, {len(segs)} segment(s), "
             f"{region}, analysed at {float(scale):.2f} scale"]
        L.append("")
        L.append("  detail = mean |luma − 3×3 box blur| over skin-masked pixels;")
        L.append("  skin = R>G>B, saturation 0.18–0.62, max channel 0.25–0.98.")
        L.append("  All three are 0–1 image units, so they are dimensionless.")
        L.append("")
        # "vs base", not "vs c1": chunk 1 can be the one that gets skipped for
        # having no skin in it, and the column would then be comparing against a
        # chunk that has no figure
        L.append("  chunk        frames     skin px     detail   vs base"
                 "      step      luma      sat")

        for idx, a, b, det, luma, sat, count in rows:
            span = f"{a}-{b - 1}"
            if det is None:
                L.append(f"  {idx:>5}  {span:>12}  {count:>10}"
                         f"     — too few skin pixels "
                         f"(floor {_floor(b - a)}), no figure reported")
                continue
            # prev is None only on the first MEASURED row, which is the baseline
            # both columns are relative to — a skipped chunk 1 must not silently
            # promote chunk 2 into a comparison against itself
            vs1 = "—" if prev is None else _pct(det, base)
            step = "—" if prev is None else _pct(det, prev)
            L.append(f"  {idx:>5}  {span:>12}  {count:>10}  {det:>9.5f}"
                     f"  {vs1:>8}  {step:>8}  {luma:>8.4f}  {sat:>7.4f}")
            prev = det

        L.append("")
        if base is not None:
            L.append(f"  baseline: the first measured segment, detail {base:.5f}.")
        if not attributed:
            L.append("  NO PLAN WIRED — the whole clip is one segment, so nothing "
                     "here can be")
            L.append("  attributed to a chunk. Wire H3 Chunk Plan's `plan` to "
                     "split it.")
        elif segs[-1][1] < n:
            L.append(f"  {n - segs[-1][1]} frame(s) past the plan's last chunk are "
                     "not covered.")
        L.append("  Report only — no threshold and no verdict; every quality "
                 "proxy this")
        L.append("  project has built has failed. The frames output is what you "
                 "judge, and")
        L.append("  it shows the exact region the numbers came from.")

        # the sheet is the analysed region at the analysed scale, so what is on
        # screen is what was measured rather than a flattering full frame
        sheet = torch.cat(frames, dim=1)
        w = int(frames[0].shape[1])
        for k in range(1, len(frames)):
            sheet[:, k * w:k * w + 2, :] = torch.tensor(
                [1.0, 0.2, 0.2], device=sheet.device, dtype=sheet.dtype)

        text = "\n".join(L)
        logging.info("H3DriftCheck: %s", L[0])
        return {"ui": {"h3char": [text]}, "result": (sheet[None], text)}


NODE_CLASS_MAPPINGS = {"H3DriftCheck": H3DriftCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DriftCheck": "H3 Drift Check"}
