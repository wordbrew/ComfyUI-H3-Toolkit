"""Where to cut a subject out of a clip, decided once for the whole clip.

Pure arithmetic on bounding boxes — no torch — so it can be tested. The nodes in
crop.py do the tensor work and call in here for the plan.

WHY CROP AT ALL
  H3 denoises every token every step, so a full 640x1120 x 345-frame render pays
  for ~71,000 video tokens to change a person occupying maybe a third of them.
  Attention cost grows faster than linearly in sequence length, so cutting the
  canvas to the subject is the only lever that changes the order of magnitude.
  Steps, schedulers and quantisation are all percentages by comparison.

WHY NOT JUST CROP EACH FRAME TO ITS OWN BOX
  Because the model would read the resulting wobble as camera motion. A per-frame
  box follows segmentation jitter, and jitter in the crop is jitter in the frame.
  Both modes here produce boxes that are stable by construction:

    static   one box for the whole clip: the padded union of every frame's
             subject. Nothing moves, so nothing can wobble. For a locked-off
             camera with a subject working a limited area — which is most of what
             we shoot — this is nearly as tight as tracking and has no failure
             mode at all.

    tracked  one constant SIZE for the whole clip, and a position that HOLDS
             STILL until the subject would leave the box, then moves the minimum
             needed. Motion is driven by a median-filtered track, so brief
             excursions and mask noise do not move the crop; only sustained
             motion does. Size is constant so slicing stays pixel-exact.

  Neither mode resamples. That is deliberate and it is the same argument as
  H3MaskInpaint's crop-don't-stretch: most of the output is the source, so any
  resampling spends sharpness on pixels that were going to be kept verbatim.
  A mode that rescales the crop to a fixed output size (so the subject stays the
  same size as it approaches camera) would resample on the way in AND on the way
  back out. It is worth that cost only when subject scale changes a lot, which
  locked-off footage does not do.

APPROXIMATION, STATED HONESTLY
  drozbay/MaskVidExperiments solves this as a linear program: box size and every
  frame's position chosen together over the whole clip, with movement, padding
  shortfall and excess area all priced, and containment as a hard constraint. That
  is a better answer than greedy hysteresis and it is where the idea came from.
  Ours fixes the size first, from the widest padded extent, and then walks the
  positions. Where that loses is a subject whose size changes a lot: the box ends
  up sized for the largest frame and is wasteful on all the others. Static mode
  has the same property and does not pretend otherwise.
"""

import math


def _lerp_gaps(track, known):
    """Fill frames where segmentation found nothing.

    A dropped frame in the middle interpolates between its neighbours; dropped
    frames at either end hold the nearest known value. Without this a single
    empty mask collapses the box to nothing and the crop jumps.
    """
    n = len(track)
    idx = [i for i in range(n) if known[i]]
    if not idx:
        raise ValueError("every frame's mask is empty — nothing to crop to")
    out = list(track)
    for i in range(n):
        if known[i]:
            continue
        if i < idx[0]:
            out[i] = track[idx[0]]
        elif i > idx[-1]:
            out[i] = track[idx[-1]]
        else:
            lo = max(j for j in idx if j < i)
            hi = min(j for j in idx if j > i)
            t = (i - lo) / (hi - lo)
            out[i] = track[lo] + t * (track[hi] - track[lo])
    return out


def _median(track, window):
    """Median filter. This is what separates sustained motion from jitter.

    A mean would let one bad frame drag the crop; a median ignores it entirely
    until half the window agrees, which is the definition we want for 'sustained'.
    """
    w = max(1, int(window)) | 1          # force odd
    if w == 1:
        return list(track)
    r, n = w // 2, len(track)
    out = []
    for i in range(n):
        seg = sorted(track[max(0, i - r):min(n, i + r + 1)])
        out.append(seg[len(seg) // 2])
    return out


def _grid(v, div, cap):
    """Round a length up to the grid, never exceeding the image."""
    hi = (int(cap) // div) * div or int(cap)
    return int(min(math.ceil(v / div) * div, hi))


def _size_for(need_w, need_h, img_w, img_h, aspect, div):
    """A legal box size: on the grid, inside the image, honouring an aspect lock.

    The lock can only be approximate — both axes have to stay on the grid — so the
    caller reports the ratio actually achieved rather than the one requested.
    """
    w = _grid(need_w, div, img_w)
    h = _grid(need_h, div, img_h)
    if aspect and aspect > 0:
        if w < h * aspect:
            w = _grid(h * aspect, div, img_w)
        else:
            h = _grid(w / aspect, div, img_h)
        # capping at the image can break the ratio the other way; give the
        # surplus axis back rather than returning a box that does not fit
        if w > h * aspect + div:
            w = max(div, min(w, (int(h * aspect) // div) * div))
        elif h > w / aspect + div:
            h = max(div, min(h, (int(w / aspect) // div) * div))
    return max(div, int(w)), max(div, int(h))


def _padded_need(x0, x1, scale):
    """Half-open padded span for one axis. scale is a multiple of the extent."""
    ext = x1 - x0 + 1
    return ext * max(1.0, float(scale))


def plan(bboxes, img_w, img_h, mode="static", crop_scale=1.5, aspect_ratio=0.0,
         divisible_by=32, smooth_window=16, hold_margin=8):
    """Plan crop boxes.

    bboxes: per frame, an inclusive (x0, y0, x1, y1) or None where the mask was
    empty. Returns (boxes, info) with one box dict {x, y, width, height} per
    frame — always per frame, even in static mode, so the uncrop has one code
    path.

    crop_scale 0 disables cropping: full frames pass through, so a workflow can
    be switched between cropped and whole-frame with one number.
    """
    n = len(bboxes)
    if n == 0:
        raise ValueError("no frames")
    div = max(1, int(divisible_by))

    if not crop_scale:                                    # 0 == disabled
        w, h = (img_w // div) * div or img_w, (img_h // div) * div or img_h
        box = {"x": 0, "y": 0, "width": int(w), "height": int(h)}
        return [dict(box) for _ in range(n)], {
            "mode": "disabled", "width": w, "height": h,
            "note": "crop_scale 0 — full frames, no crop"}

    known = [b is not None for b in bboxes]
    fill = [b if b is not None else (0, 0, 0, 0) for b in bboxes]
    x0 = _lerp_gaps([float(b[0]) for b in fill], known)
    y0 = _lerp_gaps([float(b[1]) for b in fill], known)
    x1 = _lerp_gaps([float(b[2]) for b in fill], known)
    y1 = _lerp_gaps([float(b[3]) for b in fill], known)

    if mode == "static":
        ux0, uy0 = min(x0), min(y0)
        ux1, uy1 = max(x1), max(y1)
        w, h = _size_for(_padded_need(ux0, ux1, crop_scale),
                         _padded_need(uy0, uy1, crop_scale),
                         img_w, img_h, aspect_ratio, div)
        cx, cy = (ux0 + ux1 + 1) / 2, (uy0 + uy1 + 1) / 2
        x = int(min(max(0, round(cx - w / 2)), img_w - w))
        y = int(min(max(0, round(cy - h / 2)), img_h - h))
        box = {"x": x, "y": y, "width": w, "height": h}
        boxes = [dict(box) for _ in range(n)]
        info = {"mode": "static", "width": w, "height": h, "moves": 0,
                "aspect": round(w / h, 3),
                "covered": _contains_all(boxes, x0, y0, x1, y1)}
        return boxes, info

    # tracked: one size, positions that hold still
    need_w = max(_padded_need(a, b, crop_scale) for a, b in zip(x0, x1))
    need_h = max(_padded_need(a, b, crop_scale) for a, b in zip(y0, y1))
    # the box must be able to contain the widest single frame even if padding
    # would have allowed less
    need_w = max(need_w, max(b - a + 1 for a, b in zip(x0, x1)))
    need_h = max(need_h, max(b - a + 1 for a, b in zip(y0, y1)))
    w, h = _size_for(need_w, need_h, img_w, img_h, aspect_ratio, div)

    sx0, sx1 = _median(x0, smooth_window), _median(x1, smooth_window)
    sy0, sy1 = _median(y0, smooth_window), _median(y1, smooth_window)

    xs = _walk(x0, x1, sx0, sx1, w, img_w, hold_margin)
    ys = _walk(y0, y1, sy0, sy1, h, img_h, hold_margin)

    boxes = [{"x": int(xs[i]), "y": int(ys[i]), "width": w, "height": h}
             for i in range(n)]
    moves = sum(1 for i in range(1, n) if xs[i] != xs[i - 1] or ys[i] != ys[i - 1])
    info = {"mode": "tracked", "width": w, "height": h, "moves": moves,
            "aspect": round(w / h, 3),
            "covered": _contains_all(boxes, x0, y0, x1, y1)}
    return boxes, info


def _walk(lo_track, hi_track, s_lo, s_hi, size, limit, margin):
    """One axis of the tracked plan: hold position, move only when forced.

    Containment is tested against the RAW extent — the subject must never be
    clipped, that is not negotiable. But when a move is forced, the destination
    is chosen from the SMOOTHED track, so the crop settles where the subject is
    trending rather than where one noisy frame put it. `margin` overshoots the
    move slightly so the next frame of the same motion does not force another.
    """
    n = len(lo_track)
    pos = None
    out = []
    for i in range(n):
        need_lo = hi_track[i] + 1 - size          # smallest x that still contains
        need_hi = lo_track[i]                     # largest x that still contains
        need_lo = max(0.0, min(need_lo, limit - size))
        need_hi = max(0.0, min(need_hi, limit - size))
        if need_lo > need_hi:                     # box smaller than the subject
            need_lo = need_hi = max(0.0, min((lo_track[i] + hi_track[i] + 1) / 2
                                             - size / 2, limit - size))
        if pos is None:                           # open centred on the subject
            centre = (s_lo[i] + s_hi[i] + 1) / 2 - size / 2
            pos = min(max(centre, need_lo), need_hi)
        elif pos < need_lo:
            pos = min(need_lo + margin, need_hi)
        elif pos > need_hi:
            pos = max(need_hi - margin, need_lo)
        out.append(int(round(min(max(pos, 0), limit - size))))
    return out


def _contains_all(boxes, x0, y0, x1, y1):
    """Did the plan actually keep every frame's subject inside its box?

    Reported rather than asserted: a subject wider than the image cannot be
    contained, and silently clipping it is worse than saying so.
    """
    for i, b in enumerate(boxes):
        if (x0[i] < b["x"] or y0[i] < b["y"]
                or x1[i] >= b["x"] + b["width"]
                or y1[i] >= b["y"] + b["height"]):
            return False
    return True
