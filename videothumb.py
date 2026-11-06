"""One frame out of a clip, decoded on the SERVER and cached.

WHY THIS EXISTS, measured rather than guessed
  CJ: "the preview tiles for video loading in gallery also doesnt work. slow
  and wonky and causes browser to freeze." And, the same afternoon: "the video
  load and trim still doesn't work for additional references."

  Those are one bug. His input folder holds 371 mp4s totalling 6.7 GB --
  median 11.8 MB, p90 44.9 MB. The gallery drew a thumbnail by handing each
  clip to a <video preload="metadata">, so opening it asked the browser to
  fetch metadata from dozens of multi-megabyte files at once. A browser allows
  about six connections per host, so the rest queue -- and an mp4 whose `moov`
  atom sits at the END means a range request near the tail plus the header,
  megabytes per tile. The queue starves the page's own /view images and API
  calls, which is the freeze. Making the thumbnails lazy and copying the frame
  into a <canvas> (the previous fix) removed the RETAINED decoders but not the
  fetch storm, which is why it was still slow. You cannot make this cheap in
  the browser: the work is proportional to the bytes, and the bytes are the
  problem.

  And because a clip could not be previewed, it could not be picked -- so
  loading one as a reference, and therefore trimming it, was unreachable. The
  trim UI was never broken; it was behind a gallery that would not open.

  Decoded here, a tile costs about 10 KB and one cached HTTP request, which is
  what an image tile already costs.

NO NEW DEPENDENCY. PyAV is a hard ComfyUI requirement (`av>=17.0.0` in
requirements.txt, imported at module scope by comfy_api's video types), so it
is present wherever this pack runs. No ffmpeg subprocess, no imageio.

THE CACHE IS KEYED ON WHAT WOULD CHANGE THE PICTURE -- the path, the file's
mtime and size, and the requested edge. A re-encoded clip at the same name
gets a new key rather than a stale frame, and nothing has to be invalidated by
hand.
"""
from __future__ import annotations

import hashlib
import os

#: The edge of the square the frame is fitted into. 192 is twice the 84px tile
#: the picker draws, so it stays sharp on a HiDPI screen and no larger.
DEFAULT_EDGE = 192

#: Where in the clip to take the frame from. Frame 0 of an encoded video is
#: very often black or a fade, which reads as a broken thumbnail -- the same
#: reason the browser-side version seeked before capturing.
SEEK_FRACTION = 0.1
SEEK_CAP_SECONDS = 0.5

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif")


class ThumbError(RuntimeError):
    pass


def is_video(name):
    return str(name or "").lower().endswith(VIDEO_EXTS)


def cache_name(path, edge=DEFAULT_EDGE):
    """A stable file name for this clip's thumbnail. -> str

    Includes mtime and size so a replaced file cannot serve the old frame.
    Hashed because the key holds a full path, and a path is not a filename.
    """
    try:
        st = os.stat(path)
        stamp = f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        stamp = "0-0"
    key = f"{os.path.abspath(path)}|{stamp}|{int(edge)}"
    return "h3thumb-" + hashlib.sha1(key.encode("utf-8")).hexdigest() + ".webp"


def _fit(w, h, edge):
    """Cover-crop to a square, then scale. Mirrors the CSS the tiles use."""
    side = min(int(w), int(h)) or 1
    # CLAMPED. A frame reporting 0x0 made `side` fall back to 1 and the
    # offsets go NEGATIVE, which PIL answers by padding with black rather than
    # by failing -- a thumbnail that is wrong instead of absent.
    left = max(0, (int(w) - side) // 2)
    top = max(0, (int(h) - side) // 2)
    return left, top, side, int(edge)


def probe(path):
    """(frames, seconds) for a clip, without decoding it. -> (int, float)

    The container knows its own duration, so the browser never has to open a
    clip to find out how long it is -- which is the other thing it was opening
    these files for.
    """
    import av
    with av.open(str(path)) as c:
        stream = next((s for s in c.streams if s.type == "video"), None)
        if stream is None:
            raise ThumbError("no video stream")
        seconds = 0.0
        if stream.duration is not None and stream.time_base:
            seconds = float(stream.duration * stream.time_base)
        elif c.duration:
            seconds = float(c.duration) / 1_000_000.0
        rate = stream.average_rate or stream.base_rate or 24
        frames = int(stream.frames or 0)
        if not frames and seconds:
            frames = int(round(seconds * float(rate)))
        return max(0, frames), max(0.0, seconds)


def render(path, edge=DEFAULT_EDGE):
    """One frame, as WebP bytes. -> (bytes, frames, seconds)

    Raises ThumbError with something sayable; the caller turns that into a
    status rather than a wall of traceback, because a clip that cannot be
    decoded is a normal thing to find in a folder of 371 of them.
    """
    try:
        import av
        from PIL import Image
    except Exception as exc:                                    # noqa: BLE001
        raise ThumbError(f"no decoder available: {exc}") from exc

    frames, seconds = 0, 0.0
    try:
        with av.open(str(path)) as c:
            stream = next((s for s in c.streams if s.type == "video"), None)
            if stream is None:
                raise ThumbError("no video stream")
            stream.thread_type = "AUTO"
            if stream.duration is not None and stream.time_base:
                seconds = float(stream.duration * stream.time_base)
            elif c.duration:
                seconds = float(c.duration) / 1_000_000.0
            rate = float(stream.average_rate or stream.base_rate or 24)
            frames = int(stream.frames or 0)
            if not frames and seconds:
                frames = int(round(seconds * rate))

            # SEEK, don't decode from the start. A 45 MB clip decoded linearly
            # to reach half a second in is most of the cost this is here to
            # avoid. A failed seek is not fatal -- some containers cannot --
            # so it falls through to decoding from wherever it landed.
            at = min(SEEK_CAP_SECONDS, seconds * SEEK_FRACTION) if seconds \
                else 0.0
            if at > 0 and stream.time_base:
                try:
                    c.seek(int(at / float(stream.time_base)), stream=stream)
                except Exception:                               # noqa: BLE001
                    pass

            img = None
            for frame in c.decode(stream):
                img = frame.to_image()
                break
            if img is None:
                # the seek may have landed past the last keyframe
                try:
                    c.seek(0)
                    for frame in c.decode(stream):
                        img = frame.to_image()
                        break
                except Exception:                               # noqa: BLE001
                    pass
            if img is None:
                raise ThumbError("no frame could be decoded")
    except ThumbError:
        raise
    except Exception as exc:                                    # noqa: BLE001
        raise ThumbError(str(exc) or exc.__class__.__name__) from exc

    import io
    left, top, side, e = _fit(img.width, img.height, edge)
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((e, e), Image.LANCZOS)
    buf = io.BytesIO()
    # quality 70: a 192px tile, and the difference against 90 is not visible
    # at the size it is drawn while the file is half the size.
    img.save(buf, format="WEBP", quality=70, method=4)
    return buf.getvalue(), frames, seconds


def cached(path, cache_dir, edge=DEFAULT_EDGE):
    """Render once, then serve the file. -> (full path, frames, seconds)

    `frames` is 0 when the thumbnail came from the cache: the length is
    written beside it so a cache hit does not have to reopen the clip.
    """
    name = cache_name(path, edge)
    out = os.path.join(cache_dir, name)
    meta = out + ".txt"
    if os.path.exists(out):
        frames, seconds = 0, 0.0
        try:
            with open(meta, encoding="utf-8") as fh:
                a, b = fh.read().strip().split(",", 1)
                frames, seconds = int(a), float(b)
        except Exception:                                       # noqa: BLE001
            pass
        return out, frames, seconds

    data, frames, seconds = render(path, edge)
    os.makedirs(cache_dir, exist_ok=True)
    # WRITE THEN RENAME. Two tabs opening the same gallery race on the same
    # key, and a half-written webp is a broken image that caches.
    tmp = out + f".{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, out)
    try:
        with open(meta, "w", encoding="utf-8") as fh:
            fh.write(f"{frames},{seconds:.4f}")
    except OSError:
        pass
    return out, frames, seconds
