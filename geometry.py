"""Centre-crop rectangles. Pure index arithmetic — no torch, so it can be tested.

Why crop at all, rather than resize to fit:

  crop     changes framing, leaves every retained pixel bit-identical
  scale    keeps framing, resamples every pixel
  stretch  distorts anatomy AND resamples

In an inpaint most of the output IS the source — everything outside the mask is
pinned from those exact pixels — so a resize softens the whole frame to accommodate
a region that gets regenerated anyway. And a stretch additionally hands the model a
squashed body to match, which fights everything it knows about anatomy. So: crop to
the target aspect first (free), and only rescale if the size still differs.

Both crop functions return (x0, y0, w, h) so the caller can slice, and both centre
the crop, which keeps the subject in frame for the usual case of a centred subject.

`canvas_for_megapixels` is the other direction — no source rectangle, just an area
budget and a shape — and lives here because it obeys the same 32-px rule.
"""

import math


def crop_to_multiple(w, h, multiple=32):
    """Largest centred rectangle whose sides are multiples of `multiple`.

    H3 needs both canvas dimensions on a 32 grid. Trimming up to 31 px off an edge
    costs a sliver of frame; resampling to reach the same size costs sharpness
    everywhere.
    """
    cw = (int(w) // multiple) * multiple
    ch = (int(h) // multiple) * multiple
    if cw < multiple or ch < multiple:
        raise ValueError(f"{w}x{h} has no {multiple}px-aligned crop")
    return (int(w) - cw) // 2, (int(h) - ch) // 2, cw, ch


def cover_crop(sw, sh, tw, th):
    """Largest centred rectangle in sw x sh that has tw:th's aspect ratio.

    "Cover", not "contain": the crop fills the target frame with no letterboxing,
    at the cost of trimming the long axis. Returns the source rect unchanged when
    the aspects already agree, so the common case is a no-op.
    """
    sw, sh, tw, th = int(sw), int(sh), int(tw), int(th)
    if sw * th == tw * sh:
        return 0, 0, sw, sh
    if sw * th > tw * sh:                       # source is wider -> trim the sides
        cw, ch = max(1, round(sh * tw / th)), sh
    else:                                       # source is taller -> trim top/bottom
        cw, ch = sw, max(1, round(sw * th / tw))
    return (sw - cw) // 2, (sh - ch) // 2, cw, ch


def canvas_for_megapixels(mp, aspect_w, aspect_h, multiple=32, cap_mp=0.0):
    """-> (width, height, used_mp). A render canvas from an AREA and a shape.

    Area is the thing that decides cost — H3 denoises every token every step and
    the token count is `(w/32)*(h/32)` per latent frame — so an area budget is
    the useful handle, and the aspect ratio is a separate, free choice. Same form
    as the crop detailer's upscale in crop.py: solve w*h = mp and w/h = ar, then
    round each axis to the grid independently.

    Rounding each axis moves BOTH the area and the ratio a little, so the
    delivered megapixels are rarely the ones asked for and the caller is expected
    to report what it got rather than what it wanted. Rounding to nearest (not
    down) also means a request already at `cap_mp` can land a hair above it —
    768x1344 is 1.032 MP against a 1.03 cap — which is a rounding, not an
    overshoot worth refusing.

    `cap_mp` clamps the REQUEST. 0 disables it. `used_mp` is what was actually
    solved for, so a caller can tell a clamp happened and say so.
    """
    want = float(mp)
    if cap_mp and want > float(cap_mp):
        want = float(cap_mp)
    div = max(1, int(multiple))
    ar = float(aspect_w) / float(aspect_h)
    w = max(div, int(round(math.sqrt(want * 1e6 * ar) / div)) * div)
    h = max(div, int(round(math.sqrt(want * 1e6 / ar) / div)) * div)
    return w, h, want
