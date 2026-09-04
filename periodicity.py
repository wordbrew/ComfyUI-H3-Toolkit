"""Is there structure at a fixed frame interval? Answer with a number.

WHY THIS EXISTS
  The H3 video VAE encodes in `clip_length = 17` frame blocks, disjointly, with
  no overlap and no blend -- `encode_temporal` slices, encodes each block, and
  concatenates. Its SPATIAL tiling blends with a linear crossfade; the temporal
  path does not. spacepxl raised this on 2026-09-03 as a defect.

  It may not be one. 17 is the model's NATIVE block: legal runs are 17n+5 pixel
  frames mapping to 5n+2 latent frames, and `FRAME_PER_TOKEN = (1,4,4,4,4)`
  repeats with exactly that period. Slicing there is slicing on the
  architecture's own boundary, not chopping a continuous signal arbitrarily.

  So the question is empirical and nobody has measured it. This measures it.

WHAT IT MEASURES
  The frame-to-frame difference series, folded on a period. If block boundaries
  leave an edge, the deltas at frames that ARE multiples of the period sit above
  the ones that are not. The statistic is that ratio, plus where the period's
  phase actually peaks -- a real boundary artifact peaks at one phase and stays
  there, while noise wanders.

  Reported as DATA, with no threshold and no verdict. Every quality proxy in
  this project has failed at judging; this one reports a ratio, a phase, and how
  big the effect is against the clip's own spread, and CJ's eye decides whether
  0.7 seconds apart is something he has been seeing.

WHY A CONTROL PERIOD MATTERS
  A ratio above 1 means nothing on its own: any periodicity test on real footage
  finds something, because motion is not stationary. So the same statistic is
  computed at NEARBY periods that carry no architectural meaning (16, 18, 19).
  If 17 is not clearly separated from its neighbours, there is no 17-frame
  effect -- only a clip whose motion happens to be lumpy.
"""

import logging

CATEGORY = "MiniMax H3/analysis"

VAE_CLIP_LENGTH = 17


def deltas(images):
    """Mean absolute frame-to-frame difference, one value per adjacent pair."""
    import torch
    x = images
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
        x = x[..., :3].mean(dim=-1)                 # [N, H, W] luma-ish
    d = (x[1:] - x[:-1]).abs()
    return d.flatten(1).mean(dim=1).tolist()


def fold(series, period, offset=0):
    """Mean delta at each phase of `period`. Index 0 is the boundary phase.

    `series[i]` is the difference between frame i and i+1, so a boundary
    BETWEEN frame k-1 and frame k shows up at index k-1. `offset` shifts which
    absolute frame counts as phase 0.
    """
    buckets = [[] for _ in range(period)]
    for i, v in enumerate(series):
        buckets[(i + 1 - offset) % period].append(v)
    return [sum(b) / len(b) if b else 0.0 for b in buckets]


def score(series, period, offset=0):
    """(ratio at the boundary phase, peak phase, peak ratio).

    ratio = mean delta at phase 0 / mean delta at every other phase. Above 1
    means the boundary frames move more than their neighbours do.
    """
    ph = fold(series, period, offset)
    if len(ph) < 2:
        return 1.0, 0, 1.0
    rest = [v for k, v in enumerate(ph) if k != 0]
    base = sum(rest) / len(rest)
    ratio = (ph[0] / base) if base > 0 else 1.0
    peak = max(range(len(ph)), key=lambda k: ph[k])
    rest_p = [v for k, v in enumerate(ph) if k != peak]
    base_p = sum(rest_p) / len(rest_p) if rest_p else 0.0
    peak_ratio = (ph[peak] / base_p) if base_p > 0 else 1.0
    return ratio, peak, peak_ratio


class H3Periodicity:
    """Test a clip for structure repeating at a fixed frame interval."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "A DECODED clip. The longer the "
                                            "better — the statistic is a mean "
                                            "over every boundary, so a 30s take "
                                            "has ~42 of them and a 39-frame one "
                                            "has two."}),
            "period": ("INT", {"default": VAE_CLIP_LENGTH, "min": 2, "max": 512,
                       "tooltip": "17 is H3's VAE clip_length — the block "
                                  "encode_temporal slices on, and the period "
                                  "FRAME_PER_TOKEN repeats with. That is the "
                                  "one with a reason behind it."}),
            "offset": ("INT", {"default": 0, "min": 0, "max": 511,
                       "tooltip": "Which absolute frame counts as phase 0. "
                                  "Leave at 0 unless the clip was trimmed at "
                                  "the head, which would shift every boundary."}),
        }, "optional": {
            "controls": ("STRING", {"default": "16, 18, 19, 23",
                         "tooltip": "Nearby periods with no architectural "
                                    "meaning. A ratio above 1 at 17 means "
                                    "nothing unless 17 stands clear of these — "
                                    "real footage is lumpy and every period "
                                    "finds something."}),
        }}

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("info", "ratio")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Fold the frame-to-frame difference on a period and report "
                   "whether the boundary frames move more than the rest. Data, "
                   "not a verdict.")

    def go(self, images, period, offset, controls="16, 18, 19, 23"):
        n = int(images.shape[0])
        # the length guard runs BEFORE any tensor work, so a clip that cannot
        # answer the question says so instead of failing on an import
        if n - 1 < period * 3:
            info = (f"H3 PERIODICITY — {n} frames is too short for period "
                    f"{period}. Needs at least {period * 3 + 1} to have three "
                    f"cycles to average over; fewer and the phases are single "
                    f"samples, not means.")
            logging.info("H3Periodicity: %s", info)
            return {"ui": {"h3char": [info]}, "result": (info, 1.0)}

        series = deltas(images)
        ratio, peak, peak_ratio = score(series, period, offset)
        ph = fold(series, period, offset)
        srt = sorted(series)
        med = srt[len(srt) // 2]
        p95 = srt[int(len(srt) * 0.95)]

        rows = [f"H3 PERIODICITY — {n} frames, period {period}, "
                f"{len(series) // period} full cycles",
                f"  boundary phase / all others : {ratio:.3f}x",
                f"  strongest phase             : {peak} at {peak_ratio:.3f}x"
                + ("   <- the boundary" if peak == 0 else
                   f"   <- NOT the boundary (that is phase 0)"),
                f"  clip's own deltas           : median {med:.5f}, "
                f"p95 {p95:.5f}"]

        ctrl = []
        for tok in str(controls).replace(",", " ").split():
            try:
                p = int(tok)
            except ValueError:
                continue
            if p < 2 or len(series) < p * 3:
                continue
            r, pk, _ = score(series, p, offset)
            ctrl.append((p, r, pk))
        if ctrl:
            rows.append("  control periods (no architectural meaning):")
            for p, r, pk in ctrl:
                rows.append(f"    period {p:<4} boundary ratio {r:.3f}x   "
                            f"strongest phase {pk}")
            best = max([r for _, r, _ in ctrl])
            rows.append(f"  period {period} is {ratio - best:+.3f} against the "
                        f"strongest control ({best:.3f}x)")
            if ratio <= best:
                rows.append("  -> NO effect at this period beyond what an "
                            "arbitrary one finds. The clip is lumpy, that is "
                            "all.")

        rows.append("  per-phase means (phase 0 is the boundary):")
        rows.append("    " + " ".join(f"{v:.4f}" for v in ph[:12])
                    + (" ..." if len(ph) > 12 else ""))
        rows.append("  Reported as DATA. There is no threshold here — a ratio "
                    "is not a verdict, and the strip is what you judge.")

        info = "\n".join(rows)
        logging.info("H3Periodicity: period %d ratio %.3f peak phase %d",
                     period, ratio, peak)
        return {"ui": {"h3char": [info]}, "result": (info, float(ratio))}


NODE_CLASS_MAPPINGS = {"H3Periodicity": H3Periodicity}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Periodicity": "H3 Periodicity (block seams)"}
