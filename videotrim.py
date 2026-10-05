"""Load part of a video, picked on a timeline instead of guessed in two boxes.

WHAT THIS REPLACES
  VHS's Load Video does the right work with the wrong interface. You set
  `skip_first_frames` and `frame_load_cap` as two unrelated numbers, in a frame
  domain the widgets never name, and to see what you got you wait: its preview
  endpoint shells out to ffmpeg and RE-ENCODES the trimmed range server-side on
  every change.

  Core 0.35's own `LoadVideo` shows that the re-encode is unnecessary -- it
  serves the original file and lets the browser decode it. This node keeps VHS's
  whole functional surface (rate conversion, frame selection, resizing, audio,
  frame counts) and takes core's approach to showing it.

DECODING IS CORE'S JOB, NOT OURS
  `VideoFromFile` already handles alpha, palette formats, colour space, odd
  pixel formats and audio resampling, and its `as_trimmed` seeks rather than
  decoding from zero. Reimplementing that on raw PyAV is how a loader ends up
  subtly wrong on exactly the footage you care about. So: core decodes the
  SOURCE WINDOW a span needs, and everything after that is index arithmetic on
  a tensor that is already in memory.

  The cost is honest and worth naming: rate conversion here is frame SELECTION,
  not interpolation. 30 -> 24 fps drops every fifth frame. That is also what
  VHS does (`cv_frame_generator` drops and duplicates; the ffmpeg `fps` filter
  does the same), because rate conversion without motion interpolation is
  frame selection whoever writes it.

THE SPANS WIDGET IS THE TRUTH
  `start count [label]`, one line each. The timeline in `web/` reads and writes
  that widget and nothing else, so the node works with the JavaScript disabled,
  the numbers are visible in the graph, and they land in the workflow -- and
  therefore in the metadata of every mp4 rendered from it, where they are the
  only record of which part of which file a clip came from.

  See videoframes.py for why every number here is in OUTPUT frames.
"""

import logging
import os

import torch

from . import videoframes as F

CATEGORY = "video"


def _probe(path):
    """(VideoFromFile, src_frames, src_fps, width, height) for a file.

    THE FRAME COUNT IS CROSS-CHECKED, because core's returns 1 for a whole class
    of file -- see videoframes.sane_frame_count for which files and why. Corrected
    HERE rather than in the panel so the timeline and the render agree: they both
    come through this function, and a count fixed in only one of them would draw
    a clip the node does not emit.
    """
    from comfy_api.input_impl import VideoFromFile
    video = VideoFromFile(path)
    fps = float(video.get_frame_rate())
    reported = int(video.get_frame_count())
    try:
        duration = float(video.get_duration())
    except Exception:
        duration = 0.0
    n = F.sane_frame_count(reported, duration, fps)
    if n != reported:
        logging.info("VideoTrim: %s reports %d frame(s), which is ComfyUI's "
                     "count-loop bug; %.3fs at %g fps is %d",
                     os.path.basename(str(path)), reported, duration, fps, n)
    w, h = video.get_dimensions()
    return video, n, fps, int(w), int(h)


def _resize(images, width, height, divisible_by=8):
    """Conform to a canvas, keeping the aspect when only one axis is given.

    0/0 leaves the clip alone. One axis set derives the other from the source's
    shape, which is what you want nine times in ten -- typing both is how a clip
    gets quietly stretched.
    """
    import comfy.utils
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    tw, th = int(width or 0), int(height or 0)
    if tw <= 0 and th <= 0:
        return images, w, h
    if th <= 0:
        th = round(h * tw / w)
    elif tw <= 0:
        tw = round(w * th / h)
    g = max(1, int(divisible_by or 1))
    tw = max(g, round(tw / g) * g)
    th = max(g, round(th / g) * g)
    if (tw, th) == (w, h):
        return images, w, h
    out = comfy.utils.common_upscale(images.movedim(-1, 1), tw, th,
                                     "lanczos", "disabled").movedim(1, -1)
    return out.clamp(0, 1), tw, th


def _select(images, src_fps, force_rate, every_nth, want):
    """SOURCE frames in, OUTPUT frames out. The arithmetic is in videoframes.py."""
    idx = F.select_indices(int(images.shape[0]), src_fps, force_rate,
                           every_nth, want)
    if not idx:
        return images[:0]
    return images[torch.tensor(idx, dtype=torch.long, device=images.device)]


def _trim_audio(audio, start_s, dur_s):
    """The same window of sound as of picture, sliced never resampled."""
    if not audio or audio.get("waveform") is None:
        return audio
    wav = audio["waveform"]
    sr = int(audio.get("sample_rate") or 0)
    if sr <= 0 or wav.shape[-1] == 0:
        return audio
    a = max(0, int(round(start_s * sr)))
    b = wav.shape[-1] if dur_s <= 0 else min(wav.shape[-1],
                                             a + int(round(dur_s * sr)))
    return {"waveform": wav[..., a:b], "sample_rate": sr}


def load_span(path, span, force_rate=0.0, every_nth=1, width=0, height=0,
              divisible_by=8):
    """One span of a file -> (images, audio, info dict).

    The whole point of the node in one function, so a test can drive it without
    the node wrapper and a caller can reuse it.
    """
    video, src_n, src_fps, src_w, src_h = _probe(path)
    total = F.output_length(src_n, src_fps, force_rate, every_nth)
    start, count = F.clamp_span(span.get("start", 0), span.get("count", 0), total)
    out_fps = F.output_fps(src_fps, force_rate, every_nth)

    # SEEK, do not skip. Asking core for the source window this span needs is
    # what keeps a 10-second trim out of a 40-minute file cheap; decoding from
    # zero and throwing frames away is the thing being replaced.
    t0 = F.source_time(start, src_fps, force_rate, every_nth)
    t1 = F.source_time(start + count, src_fps, force_rate, every_nth)
    dur = max(0.0, t1 - t0)
    window = video.as_trimmed(t0, dur, strict_duration=False) or video
    comp = window.get_components()

    images = _select(comp.images, src_fps, force_rate, every_nth, count)
    images, w, h = _resize(images, width, height, divisible_by)
    audio = _trim_audio(comp.audio, 0.0, dur)

    info = {"source_frames": src_n, "source_fps": src_fps,
            "source_width": src_w, "source_height": src_h,
            "output_total": total, "fps": out_fps,
            "start": start, "count": int(images.shape[0]),
            "width": w, "height": h,
            "duration": (int(images.shape[0]) / out_fps) if out_fps else 0.0,
            "label": span.get("label", "")}
    return images, audio, info


class VideoTrimLoad:
    """Load a video and pass on the part of it you marked."""

    @classmethod
    def INPUT_TYPES(cls):
        # THE WHOLE PROBE IS GUARDED, not just the listdir. INPUT_TYPES runs at
        # REGISTRATION, so anything that raises here takes the node out of the
        # menu entirely and reports it as an import failure with no mention of
        # this line. An empty file list is a node you can still see and wire.
        try:
            import folder_paths
            d = folder_paths.get_input_directory()
            files = folder_paths.filter_files_content_types(
                [f for f in os.listdir(d)
                 if os.path.isfile(os.path.join(d, f))], ["video"])
        except Exception:                       # pragma: no cover - no ComfyUI
            files = []
        return {"required": {
            # NO `video_upload: True`, DELIBERATELY.
            #
            # That key is what draws ComfyUI's "choose file to upload" button,
            # and the button is drawn on the LiteGraph CANVAS at this widget's
            # slot -- while the timeline is a DOM element positioned over the
            # node. The panel covers the button, so clicking it does nothing at
            # all, which is worse than not offering it. Un-covering it would
            # mean predicting the frontend's layout, which is not something to
            # depend on.
            #
            # Uploading lives in the panel instead: a "Choose video" label and
            # a drop target over the whole panel, both hitting /upload/image.
            # One door, and it is one you can reach.
            "file": (sorted(files), {
                     "tooltip": "Pick a video already in the input folder — or "
                                "use Choose video on the panel below, or drop "
                                "one onto it.\n\nThe preview is the file "
                                "itself, decoded by the browser; nothing is "
                                "re-encoded to show it to you."}),
            "spans": ("STRING", {"multiline": True, "default": "0 0",
                      "tooltip": "One span per line: `start count [label]`, in "
                                 "OUTPUT frames — after force_rate and "
                                 "select_every_nth. A count of 0 runs to the "
                                 "end.\n\nThe timeline writes this for you; it "
                                 "is a text widget so it survives a save, reads "
                                 "in a diff, and records in the finished mp4's "
                                 "metadata which part of which file a clip came "
                                 "from."}),
            "span_index": ("INT", {"default": 0, "min": 0, "max": 999,
                           "tooltip": "WHICH marked span this run emits, "
                                      "counting from 0. Only matters once you "
                                      "have marked more than one; the timeline "
                                      "sets it when you pick from the list, and "
                                      "hides it while it is on screen."}),
            "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0,
                           "step": 0.01,
                           "tooltip": "Resample to this frame rate, so a 30fps "
                                      "clip can feed a model that generates at "
                                      "24. 0 keeps the source's own.\n\nSelection, not "
                                      "interpolation: 30 -> 24 drops every fifth "
                                      "frame. Set it to the rate your model "
                                      "generates at and the frame counts on the "
                                      "timeline become the counts it will see."}),
            "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100,
                                 "tooltip": "Keep one frame in N. This DROPS "
                                            "frames, so the output rate falls "
                                            "with it and the clip keeps its "
                                            "real-time duration."}),
        }, "optional": {
            "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                      "tooltip": "0 = leave it alone. Set ONE axis and the "
                                 "other follows the source's aspect; setting "
                                 "both is how a clip gets stretched."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                       "tooltip": "0 = leave it alone. Set this ALONE and the "
                                  "width follows the source's aspect; setting "
                                  "both is how a clip gets stretched. Steps by 8, "
                                  "not 32 — these two loaders are "
                                  "model-agnostic and do not enforce H3's grid, "
                                  "which is H3 Match Source's job."}),
            "divisible_by": ("INT", {"default": 8, "min": 1, "max": 64,
                             "tooltip": "Round the resize to a multiple of "
                                        "this. Most video models want 8 or 32."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_count", "fps", "width", "height",
                    "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Load a video and emit ONE marked span — the one span_index "
                   "picks. Mark spans by dragging on the timeline. Keeps VHS's "
                   "rate conversion, frame selection and resizing, and previews "
                   "without re-encoding anything.")

    @classmethod
    def IS_CHANGED(cls, file, **kwargs):
        # the file's mtime, not its hash: these are large and re-reading one to
        # notice it has not changed is the slow thing core stopped doing
        import folder_paths
        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(file))
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(file):
            return f"Video file not found: {file}"
        return True

    def go(self, file, spans, span_index, force_rate, select_every_nth,
           width=0, height=0, divisible_by=8):
        import folder_paths
        path = folder_paths.get_annotated_filepath(file)
        _, src_n, src_fps, _, _ = _probe(path)
        total = F.output_length(src_n, src_fps, force_rate, select_every_nth)

        marked = F.parse_spans(spans, total)
        i = max(0, min(int(span_index), len(marked) - 1))
        images, audio, info = load_span(path, marked[i], force_rate,
                                        select_every_nth, width, height,
                                        divisible_by)

        rows = [F.describe(src_n, src_fps, force_rate, select_every_nth,
                           marked[i])]
        if len(marked) > 1:
            rows.append(f"  span {i + 1} of {len(marked)}"
                        + (f" — {info['label']}" if info["label"] else ""))
            rows += [f"    {j + 1}. {s['start']} +{s['count']}"
                     + (f"  {s['label']}" if s["label"] else "")
                     + ("   <- this run" if j == i else "")
                     for j, s in enumerate(marked)]
        if int(span_index) >= len(marked):
            rows.append(f"  span_index {int(span_index)} is past the last span; "
                        f"using {i}")
        if info["width"] != info["source_width"] or \
                info["height"] != info["source_height"]:
            rows.append(f"  resized {info['source_width']}x"
                        f"{info['source_height']} -> {info['width']}x"
                        f"{info['height']}")
        if audio is None or audio.get("waveform") is None:
            rows.append("  no audio track in this file")
        text = "\n".join(rows)
        logging.info("VideoTrimLoad: %s", rows[0])
        return {"ui": {"videotrim": [text]},
                "result": (images, audio, int(info["count"]),
                           float(info["fps"]), int(info["width"]),
                           int(info["height"]), text)}


class VideoTrimLoadAll:
    """Every span you marked, as a batch — the graph below runs once per span.

    WHY THIS IS A SECOND NODE AND NOT A CHECKBOX
      `OUTPUT_IS_LIST` is read by ComfyUI when the node is REGISTERED, not when
      it runs, so it cannot be toggled by a widget. That is just as well: it
      changes how the entire downstream graph executes, and a node that quietly
      does that depending on a checkbox is a node you cannot reason about from
      the graph. Choosing this one is the declaration.

    WHAT IT COSTS, PLAINLY
      Every node below this runs once per span, inside a single queue item. Four
      spans feeding a video model is four full renders and you cannot watch the
      first before the fourth starts. That is right for building a dataset and
      usually wrong for choosing a shot -- which is why `Load Video (trim
      timeline)` and its `span_index` stays the default way to work.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = VideoTrimLoad.INPUT_TYPES()
        req = {k: v for k, v in base["required"].items() if k != "span_index"}
        req["spans"] = (req["spans"][0], dict(req["spans"][1], tooltip=
            "One span per line: `start count [label]`. ALL of them are emitted, "
            "and everything downstream runs once per span."))
        return {"required": req, "optional": base["optional"]}

    RETURN_TYPES = VideoTrimLoad.RETURN_TYPES
    RETURN_NAMES = VideoTrimLoad.RETURN_NAMES
    OUTPUT_IS_LIST = (True,) * len(RETURN_TYPES)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Emit every marked span. Everything downstream runs once per "
                   "span, in one queue item — a dataset pass, not a shot hunt.")

    IS_CHANGED = VideoTrimLoad.IS_CHANGED
    VALIDATE_INPUTS = VideoTrimLoad.VALIDATE_INPUTS

    def go(self, file, spans, force_rate, select_every_nth, width=0, height=0,
           divisible_by=8):
        import folder_paths
        path = folder_paths.get_annotated_filepath(file)
        _, src_n, src_fps, _, _ = _probe(path)
        total = F.output_length(src_n, src_fps, force_rate, select_every_nth)
        marked = F.parse_spans(spans, total)

        cols, rows = [], []
        for i, sp in enumerate(marked):
            images, audio, info = load_span(path, sp, force_rate,
                                            select_every_nth, width, height,
                                            divisible_by)
            cols.append((images, audio, int(info["count"]), float(info["fps"]),
                         int(info["width"]), int(info["height"]), ""))
            rows.append(f"  {i + 1}. {info['start']} +{info['count']}"
                        + (f"  {info['label']}" if info["label"] else ""))
        head = (f"VIDEO TRIM: {len(marked)} span(s), "
                f"{sum(c[2] for c in cols)} frames total — everything below "
                f"this runs {len(marked)} time(s)")
        text = "\n".join([head] + rows)
        logging.info("VideoTrimLoadAll: %s", head)
        out = list(zip(*cols))
        out[-1] = [text] * len(marked)     # the same report on every branch
        return {"ui": {"videotrim": [text]}, "result": tuple(out)}


NODE_CLASS_MAPPINGS = {"VideoTrimLoad": VideoTrimLoad,
                       "VideoTrimLoadAll": VideoTrimLoadAll}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTrimLoad": "Load Video (trim timeline)",
    "VideoTrimLoadAll": "Load Video (all spans)",
}
