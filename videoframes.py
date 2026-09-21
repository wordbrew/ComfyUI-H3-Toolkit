"""The frame arithmetic, with no torch, no av and no ComfyUI in it.

WHY THIS IS ITS OWN MODULE
  Everything that can be quietly wrong about a video loader is in here. Rate
  conversion, frame selection and trimming all change what "frame 192" means,
  and they compose in an order that is invisible from the outside. A clip that
  comes out the wrong length does not error -- it renders, and the mistake shows
  up somewhere else entirely.

  So the arithmetic lives in a pure module that a test can exercise without
  ComfyUI, a video file, or a GPU.

THE ONE DECISION EVERYTHING ELSE FOLLOWS FROM
  There are three frame domains, not one:

    SOURCE   what is in the file:            src_frames at src_fps
    RATE     after force_rate:               rate_frames at rate_fps
    OUTPUT   after select_every_nth:         out_frames at out_fps

  A 30 fps, 300-frame source (10 s) at force_rate 24 is 240 RATE frames; at
  select_every_nth 2 it is 120 OUTPUT frames.

  **The timeline, the trim numbers and everything the user sees are in OUTPUT
  frames.** That is what the node emits and what downstream nodes count, so it
  is the only domain in which "192 frames" is a promise the node can keep. A
  ruler drawn in source frames is a ruler that lies as soon as force_rate is
  not the source's own rate.

  VHS exposes `skip_first_frames` and `frame_load_cap` and leaves you to work
  out which domain each is in. This module converts once, here, and the rest of
  the node never sees a source frame number.
"""

import re

# `select_every_nth` of 0 and 1 both mean "every frame". VHS treats 0 as 1 in
# some paths and divides by it in others; taking them as the same thing here
# means no caller has to remember which.
MIN_STEP = 1


def rate_of(src_fps, force_rate=0.0):
    """The frame rate after conversion. 0 means keep the source's own."""
    r = float(force_rate or 0.0)
    return r if r > 0 else float(src_fps or 0.0)


def rate_length(src_frames, src_fps, force_rate=0.0):
    """SOURCE frames -> RATE frames.

    Converted through DURATION rather than by scaling the count, because the
    count is what is being resampled: 300 frames at 30 fps is 10 s, and 10 s at
    24 fps is 240 frames. Scaling 300 by 24/30 happens to agree here and stops
    agreeing the moment the source rate is fractional (29.97, 23.976), which is
    most real footage.
    """
    src_frames = max(0, int(src_frames or 0))
    src_fps = float(src_fps or 0.0)
    if src_frames <= 0 or src_fps <= 0:
        return 0
    r = rate_of(src_fps, force_rate)
    if r <= 0 or r == src_fps:
        return src_frames
    return int(src_frames / src_fps * r)


def output_length(src_frames, src_fps, force_rate=0.0, every_nth=1):
    """SOURCE frames -> OUTPUT frames, the number the timeline is drawn in.

    Floor, not round: taking every 2nd frame of 5 yields frames 0 and 2 and 4 --
    three of them -- so the count is ceil(5/2). Half a frame is not a frame.
    """
    n = rate_length(src_frames, src_fps, force_rate)
    step = max(MIN_STEP, int(every_nth or 1))
    return -(-n // step) if n > 0 else 0


def output_fps(src_fps, force_rate=0.0, every_nth=1):
    """The rate the OUTPUT plays at.

    `select_every_nth` does not resample, it DROPS -- so the frames that survive
    are further apart in time and the clip plays faster unless the rate drops
    with them. Reporting rate/step is what makes a downstream 'frames / fps'
    duration come out right.
    """
    return rate_of(src_fps, force_rate) / max(MIN_STEP, int(every_nth or 1))


def source_time(out_index, src_fps, force_rate=0.0, every_nth=1):
    """OUTPUT frame index -> the time in the SOURCE file to seek to.

    The only conversion that touches the file, and the reason the rest of the
    node can stay in output frames.
    """
    r = rate_of(src_fps, force_rate)
    if r <= 0:
        return 0.0
    step = max(MIN_STEP, int(every_nth or 1))
    return (max(0, int(out_index)) * step) / r


def select_indices(window_frames, src_fps, force_rate=0.0, every_nth=1,
                   want=0):
    """Which frames of a decoded SOURCE window make up the output span.

    Rate conversion and frame selection are one step, because they are the same
    operation: output frame i is rate frame i*step, which is source frame
    round(i*step * src_fps/rate_fps) counted from the window's start.

    NEAREST NEIGHBOUR, and duplicates are kept. Converting UP repeats frames --
    24 -> 30 fps shows every fifth frame twice -- and that is what the model
    will see, so de-duplicating would quietly shorten the clip.

    ROUND HALF UP, not python's round(). `round()` is banker's rounding:
    round(2.5) is 2 and round(3.5) is 4. Converting 30 -> 24 fps lands exactly
    on .5 every fourth output frame, so the built-in would pick the earlier
    source frame half the time and the later one the other half -- a stutter
    with no cause you could find by reading the code. `int(x + 0.5)` ties
    upward every time, which is also what ffmpeg's fps filter does.
    """
    n = max(0, int(window_frames or 0))
    want = max(0, int(want or 0))
    if n <= 0 or want <= 0:
        return []
    rate = rate_of(src_fps, force_rate)
    step = max(MIN_STEP, int(every_nth or 1))
    ratio = (float(src_fps) / rate) if rate > 0 else 1.0
    return [min(n - 1, int(i * step * ratio + 0.5)) for i in range(want)]


def clamp_span(start, count, out_total):
    """A span inside the clip. -> (start, count), count 0 when there is no room.

    CLAMPS RATHER THAN REFUSING. A span is dragged on a timeline and typed into
    a box, and both produce out-of-range values constantly -- a loader that
    errors on one is a loader you fight. What it must never do is silently
    return a DIFFERENT span: start is pinned inside the clip and the count is
    cut to what is actually left, which is a shorter version of what was asked
    for, never a shifted one.
    """
    out_total = max(0, int(out_total or 0))
    if out_total <= 0:
        return 0, 0
    start = max(0, min(int(start or 0), out_total - 1))
    count = int(count or 0)
    if count <= 0:                      # 0 means "to the end", as in VHS
        count = out_total - start
    return start, max(0, min(count, out_total - start))


def sane_frame_count(reported, duration, fps):
    """Core's frame count, corrected when it is the broken one.

    COMFYUI'S `get_frame_count()` RETURNS 1 FOR A WHOLE CLASS OF FILE, and this
    is the bug behind a trim that collapsed to a single frame and would not be
    typed out of (CJ, 2026-09-13, on two different 3-second 30 fps clips).

    Read comfy_api/latest/_input_impl/video_types.py. It tries three things:

      1. `video_stream.frames` -- absent (0) in webm, mkv and plenty of mp4s
         that were remuxed or came off a phone;
      2. stream `duration` x `average_rate` -- skipped when the STREAM carries
         no duration, which is common in the same files;
      3. decode and count. Which is where it goes wrong:

             start_time, duration = self.get_active_trim_window()   # (0.0, 0.0)
             end_pts = int((start_time + duration) / time_base)     # -> 0
             ...
             for frame in frame_iterator:
                 if frame.pts >= end_pts: break                     # immediately
             return frame_count                                     # 1

         `duration` defaults to 0 and means "no limit" everywhere else in the
         class, but here it becomes an end timestamp of 0, so the count loop
         breaks on its first frame. Untrimmed, branch 3 ALWAYS returns 1.

    `get_duration()` does not share the fault -- it reads the CONTAINER duration
    and has real fallbacks -- so duration x fps is the cross-check.

    Only a reported count of 1 is overridden, because that is the exact
    signature of branch 3, and only when the duration disagrees. A genuinely
    one-frame file has a duration that agrees, and is left alone.
    """
    n = int(reported or 0)
    fps = float(fps or 0.0)
    dur = float(duration or 0.0)
    if n > 1 or fps <= 0 or dur <= 0:
        return max(0, n)
    est = int(round(dur * fps))
    return est if est > 1 else max(0, n)


def refit_span(start, count, out_total):
    """A span whose CLIP changed under it. -> (start, count), length-first.

    NOT clamp_span, and the difference is the whole point.

    `clamp_span` is for one EDIT: you typed a start, so the start is honoured
    and the count is cut to whatever is left. That is right when you are the one
    who moved it.

    This is for when the clip changed and the span did not -- loading the next
    file, or changing force_rate. There, the LENGTH is what you chose and the
    position is incidental, so the length is kept and the start slides back to
    make room. It is the same behaviour as dragging the block against the end of
    the timeline, which is where the habit already is.

    THE BUG THIS FIXES (2026-09-13): stepping through a folder with a 120-frame
    span set, `clamp_span` hit a clip shorter than the span's START, pinned the
    start to the last frame and cut the count to 1 -- and then WROTE THAT BACK,
    so every later clip inherited a 1-frame span that looked unchangeable.
    Clamping is lossy, and lossy is fine for an edit you can see and undo; it is
    not fine for something that happens because you clicked next.
    """
    out_total = max(0, int(out_total or 0))
    if out_total <= 0:
        return max(0, int(start or 0)), max(0, int(count or 0))
    count = int(count or 0)
    if count <= 0:
        count = out_total
    count = min(count, out_total)
    start = max(0, min(int(start or 0), out_total - count))
    return start, count


_SPAN = re.compile(r"^\s*(\d+)\s+(\d+)\s*(.*)$")


def parse_spans(text, out_total=0):
    """The spans widget -> [{'start', 'count', 'label'}], always at least one.

    `start count [label]`, one per line, `#` comments. NOT json: this is a
    widget a person edits by hand before the timeline exists to draw it, it has
    to survive a diff, and it lands in the workflow -- and therefore in every
    rendered mp4's metadata -- where it is the only record of which part of
    which file a clip came from.

    A file with no spans yields one covering everything, so a freshly dropped
    video is immediately usable.
    """
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SPAN.match(line)
        if not m:
            continue
        start, count = int(m.group(1)), int(m.group(2))
        if out_total:
            start, count = clamp_span(start, count, out_total)
        out.append({"start": start, "count": count,
                    "label": m.group(3).strip()})
    if not out:
        start, count = clamp_span(0, 0, out_total) if out_total else (0, 0)
        out.append({"start": start, "count": count, "label": ""})
    return out


def format_spans(spans):
    """The inverse, for the timeline to write back."""
    rows = []
    for s in spans or []:
        row = f"{int(s.get('start', 0))} {int(s.get('count', 0))}"
        label = (s.get("label") or "").strip()
        rows.append(f"{row} {label}".rstrip())
    return "\n".join(rows)


def describe(src_frames, src_fps, force_rate=0.0, every_nth=1, span=None):
    """One line saying what will actually come out, in both clocks."""
    total = output_length(src_frames, src_fps, force_rate, every_nth)
    fps = output_fps(src_fps, force_rate, every_nth)
    start, count = clamp_span((span or {}).get("start", 0),
                              (span or {}).get("count", 0), total)
    secs = count / fps if fps > 0 else 0.0
    src_secs = (src_frames / src_fps) if src_fps else 0.0
    bits = [f"source {src_frames} frames @ {src_fps:g} fps ({src_secs:.2f}s)"]
    if fps != float(src_fps or 0):
        bits.append(f"output {total} frames @ {fps:g} fps")
    bits.append(f"span {start}-{start + count} = {count} frames ({secs:.2f}s)")
    return "  |  ".join(bits)
