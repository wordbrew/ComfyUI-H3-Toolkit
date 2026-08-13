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

Both functions return (x0, y0, w, h) so the caller can slice, and both centre the
crop, which keeps the subject in frame for the usual case of a centred subject.
"""


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
