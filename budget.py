"""Reference token accounting for MiniMax-H3.

WHY THIS EXISTS
  A reference image contributes a FIXED number of tokens regardless of clip
  length: one latent frame's worth, `(w/32) * (h/32)`. The target video
  contributes `latent_t * (w/32) * (h/32)`, which grows linearly with duration.
  So the reference's share of the picture tokens falls as the clip gets longer,
  and the source content — which lives INSIDE the target video stream as the
  pinned region plus the `(1-sigma)` residual — scales with it instead.

  That ratio is invisible in the graph. It is not a widget, it is not in any
  log line, and nothing warns you when it collapses. You find out after the
  render. This node makes it a number you read before queueing.

  Measured 2026-08-22, character swap, denoise 0.45, 3 references, `max`:

      39 frames    ref share 19.4%   replacement character's black hair
     300 frames    ref share  3.2%   source character's blonde hair

  Two points, one clip, one subject. That is DATA, not a threshold — where the
  crossover sits, whether it is sharp or gradual, and whether it moves with
  subject or denoise are all unmeasured. This node reports the number and shows
  you where the levers move it. It does not tell you what is good.

SIZING COMES FROM CORE
  `MiniMaxH3ReferenceToVideo` lives in ComfyUI itself (`comfy_extras/
  nodes_minimax_h3.py`) and does the resizing this node has to PREDICT. The
  constants are imported from there rather than copied, so a change to
  `REF_IMAGE_SHORT_EDGE` follows automatically. The formulae are reimplemented
  and would NOT follow a change; `notes` says so in the report itself, because
  a reporting node that lies quietly is worse than no node at all.
"""

import math

CATEGORY = "MiniMax H3/video"

# Stock MiniMaxH3ReferenceToVideo declares these as Autogrow lists; a plain node
# cannot autogrow, so the ceilings are mirrored as fixed optional slots.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3

# Fallbacks matching ComfyUI v0.33.2. Used only if core moves the module —
# the report says which set is in play.
_FALLBACK = {"CANVAS_MULTIPLE": 32, "REF_IMAGE_SHORT_EDGE": 2048,
             "BASE_SHORT_EDGE": 768, "MAX_PIXELS": 768 * 1344}


def _core_constants():
    """(constants, source_label). Never raises."""
    try:
        import comfy_extras.nodes_minimax_h3 as core
        got = {k: getattr(core, k) for k in _FALLBACK if hasattr(core, k)}
        if len(got) == len(_FALLBACK):
            return got, "imported from comfy_extras.nodes_minimax_h3"
        missing = sorted(set(_FALLBACK) - set(got))
        merged = dict(_FALLBACK)
        merged.update(got)
        return merged, "partly imported; fell back for " + ", ".join(missing)
    except Exception as exc:
        return dict(_FALLBACK), "core import failed (%s) — using v0.33.2 fallbacks" % type(exc).__name__


def _core_adapt_canvas():
    """core's adapt_canvas, or None. Reference VIDEOS need it; images do not."""
    try:
        from comfy_extras.nodes_minimax_h3 import adapt_canvas
        return adapt_canvas
    except Exception:
        return None


def frame_rows(px_w, px_h):
    """Tokens in ONE latent frame: the DiT's 2x2 patch grid over the /16 latent.

    Identical to `chunkplan.tokens_per_frame`, which writes the same number as
    `(w/32)*(h/32)` from the canvas side. Not imported from there: this module is
    loaded as a flat module by test_budget.py, so a top-level relative import
    would break the torch-free test path.
    """
    return (int(px_w) // 16 // 2) * (int(px_h) // 16 // 2)


def ref_image_encoded_size(w, h, gen_w, gen_h, mode, C):
    """Predict what core resizes a reference image to. Mirrors nodes_minimax_h3."""
    cm = C["CANVAS_MULTIPLE"]
    if mode == "match":
        scale = min(1.0, math.sqrt((gen_w * gen_h) / float(w * h)))
    else:
        scale = min(1.0, C["REF_IMAGE_SHORT_EDGE"] / float(min(w, h)))
    tw = max(cm, round(w * scale / cm) * cm)
    th = max(cm, round(h * scale / cm) * cm)
    return int(tw), int(th)


def _bar(pct, width=28):
    filled = max(0, min(width, int(round(pct / 100.0 * width))))
    return "[" + "#" * filled + "." * (width - filled) + "]"


class H3RefBudget:
    """Report how much of the sequence your references actually occupy.

    Wire the SAME `width` / `height` / `length` / `ref_image_size` as the
    reference node, and the same images, and it reports what those will encode
    to and what share of the picture tokens they will hold.

    `info` is a plain STRING and connects straight to rgthree's Display Any.
    `ref_share` (0-1) and `seq_len` are there so a graph can branch on them.

    Reference VIDEOS are included when core's `adapt_canvas` is importable —
    they use a different sizing path (a 768-short-edge canvas, not the
    match/max rule) and their token count also depends on the 17n+5 frame
    snapping, both of which this mirrors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "width": ("INT", {"default": 640, "min": 32, "max": 4096, "step": 32,
                              "tooltip": "The generation width, not the reference's."}),
            "height": ("INT", {"default": 1120, "min": 32, "max": 4096, "step": 32}),
            "length": ("INT", {"default": 294, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "Frames at 24 fps. Snapped to the 17n+5 grid "
                                          "for the report, the same way the reference "
                                          "node snaps it."}),
            "ref_image_size": (["match", "max"], {"default": "max",
                               "tooltip": "'match' shrinks each reference to the "
                                          "generation's pixel AREA. 'max' leaves it "
                                          "native up to a 2048 short edge, which on a "
                                          "large source is far more tokens. Must match "
                                          "the reference node's setting or the report "
                                          "describes a render you are not doing."}),
        }
        # stock MiniMaxH3ReferenceToVideo autogrows ref_image_1..9 and
        # ref_video_1..3, so match its ceiling rather than our own node's 5
        opt = {f"ref_image_{i}": ("IMAGE", {"tooltip": "Same image you feed the "
                                                       "reference node."})
               for i in range(1, MAX_REF_IMAGES + 1)}
        opt.update({f"ref_video_{i}": ("IMAGE", {"tooltip": "A reference VIDEO, as a "
                                                            "frame batch. Sized by "
                                                            "core's canvas rule, not "
                                                            "by ref_image_size."})
                    for i in range(1, MAX_REF_VIDEOS + 1)})
        opt["prompt"] = ("STRING", {"multiline": True, "forceInput": True,
                                    "tooltip": "Optional, only to estimate the text "
                                               "rows in seq_len. The share figures are "
                                               "picture-only and do not use it."})
        return {"required": req, "optional": opt}

    # ref_tokens is APPENDED, never inserted -- saved workflows store slot
    # indices, so a new output goes on the end or every existing link shifts.
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("info", "ref_share", "seq_len", "ref_tokens")
    OUTPUT_TOOLTIPS = ("The full budget report.",
                       "References as a fraction of the PICTURE tokens, 0-1.",
                       "The WHOLE packed sequence: video + audio + references + "
                       "text. Not a reference count — wiring this where a "
                       "reference count is wanted reports a nonsense share.",
                       "Just the reference rows. This is what H3 Chunk Plan's "
                       "`ref_tokens` wants, so it can report each chunk's share "
                       "as its length changes.")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    OUTPUT_NODE = False
    DESCRIPTION = ("How many tokens your references hold versus the target video, "
                   "and how that share moves with clip length. Reports the number; "
                   "does not judge it.")

    def go(self, width, height, length, ref_image_size, **kw):
        from .timing import video_latent_t, snap_run, audio_t, is_av_aligned

        C, const_src = _core_constants()
        adapt_canvas = _core_adapt_canvas()

        frames = snap_run(length)
        lt = video_latent_t(frames)
        at = audio_t(frames)
        tgt_rows = frame_rows(width, height)
        video_tok = lt * tgt_rows
        audio_rows = at * 2

        L = []
        L.append("H3 REFERENCE BUDGET")
        L.append("=" * 62)
        L.append("generation   %dx%d  %d frames (%.2fs)" % (width, height, frames,
                                                            frames / 24.0))
        if frames != int(length):
            L.append("             NOTE length %d is off the 17n+5 grid, snapped to %d"
                     % (int(length), frames))
        if not is_av_aligned(frames):
            L.append("             NOTE %d is not AV-aligned; the audio clock rounds here"
                     % frames)
        L.append("             %d latent frames x %d tok/frame = %s video tokens"
                 % (lt, tgt_rows, format(video_tok, ",")))
        L.append("")

        rows = []
        ref_tok = 0
        for i in range(1, MAX_REF_IMAGES + 1):
            img = kw.get(f"ref_image_{i}")
            if img is None:
                continue
            h, w = int(img.shape[1]), int(img.shape[2])
            tw, th = ref_image_encoded_size(w, h, width, height, ref_image_size, C)
            n = frame_rows(tw, th)
            ref_tok += n
            note = ""
            if ref_image_size == "max" and min(w, h) > C["REF_IMAGE_SHORT_EDGE"]:
                note = "  (capped at short edge %d)" % C["REF_IMAGE_SHORT_EDGE"]
            elif ref_image_size == "match" and (tw * th) < (w * h):
                note = "  (shrunk to the generation's area)"
            rows.append("ref_image_%d  %dx%d -> %dx%d  %s tok%s"
                        % (i, w, h, tw, th, format(n, ">7,"), note))

        for vi in range(1, MAX_REF_VIDEOS + 1):
            vid = kw.get(f"ref_video_{vi}")
            if vid is None:
                continue
            vh, vw = int(vid.shape[1]), int(vid.shape[2])
            nframes = int(vid.shape[0])
            if adapt_canvas is None:
                rows.append("ref_video_%d  %dx%d x%df -> NOT COUNTED (core adapt_canvas "
                            "unavailable)" % (vi, vw, vh, nframes))
                continue
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:  # core keeps a small source at its own size
                cm = C["CANVAS_MULTIPLE"]
                cw = max(cm, round(vw / cm) * cm)
                ch = max(cm, round(vh / cm) * cm)
            n_used = min(nframes, frames)
            while n_used % 17 != 5 and n_used > 5:
                n_used -= 1
            if n_used < 5:
                rows.append("ref_video_%d  %dx%d x%df -> REJECTED, needs >=5 frames"
                            % (vi, vw, vh, nframes))
                continue
            vlt = video_latent_t(n_used)
            n = vlt * frame_rows(cw, ch)
            ref_tok += n
            drop = "" if n_used == nframes else (
                "  (%d frames dropped to the 17n+5 grid)" % (nframes - n_used))
            rows.append("ref_video_%d  %dx%d x%df -> %dx%d x%d latent  %s tok%s"
                        % (vi, vw, vh, nframes, cw, ch, vlt, format(n, ">7,"), drop))

        if not rows:
            L.append("no references connected — nothing to weigh")
            L.append("")
        else:
            L.extend(rows)
            L.append("-" * 62)

        picture = ref_tok + video_tok
        share = (ref_tok / float(picture)) if picture else 0.0
        if rows:
            L.append("references   %s tok        %5.1f%% of picture tokens"
                     % (format(ref_tok, ">7,"), share * 100))
            L.append("             %s" % _bar(share * 100))
            L.append("")

        # where the share goes with length, everything else held
        if ref_tok:
            L.append("same references at other lengths:")
            for f in (39, 90, 141, 192, 243, 294, 345):
                s = ref_tok / float(ref_tok + video_latent_t(f) * tgt_rows)
                mark = "  <- this render" if f == frames else ""
                L.append("   %3df  %5.1f%%%s" % (f, s * 100, mark))
            L.append("")
            # what it would take to hold THIS share at other lengths
            if frames > 39:
                need = ref_tok * (video_latent_t(frames) / float(video_latent_t(39)))
                L.append("to hold a 39-frame ratio here you would need ~%sx the reference "
                         "tokens" % format(round(video_latent_t(frames) / float(video_latent_t(39)), 1)))
                L.append("   (%s tok, vs the %s you have)" % (format(int(need), ","),
                                                              format(ref_tok, ",")))
                L.append("")

        text_tok = 0
        prompt = kw.get("prompt")
        if prompt:
            text_tok = max(1, len(str(prompt)) // 4)  # rough, English prose
        seq = picture + audio_rows + text_tok
        L.append("sequence     video %s + audio %s + refs %s%s"
                 % (format(video_tok, ","), format(audio_rows, ","),
                    format(ref_tok, ","),
                    (" + text ~%s" % format(text_tok, ",")) if text_tok else ""))
        L.append("             = %s rows%s" % (format(seq, ","),
                 "" if text_tok else "  (text rows not counted — wire `prompt` to include them)"))
        L.append("             attention cost goes as roughly the square of this")
        L.append("")
        L.append("reference points (n=1, character swap, denoise 0.45, 3 refs, max):")
        L.append("    39f @ 19.4% -> reference's hair colour held")
        L.append("   300f @  3.2% -> source's hair colour won")
        L.append("   the crossover between them is UNMEASURED. This is data, not a rule.")
        L.append("")
        L.append("notes: constants %s." % const_src)
        L.append("       Resize FORMULAE are reimplemented from core and do not track")
        L.append("       changes to it. If these numbers stop matching your renders,")
        L.append("       re-read comfy_extras/nodes_minimax_h3.py.")
        L.append("       ref_image_size here must match the reference node's widget.")

        return ("\n".join(L), float(share), int(seq), int(ref_tok))


NODE_CLASS_MAPPINGS = {"H3RefBudget": H3RefBudget}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefBudget": "H3 Reference Budget"}
