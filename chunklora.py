"""Per-chunk LoRA scheduling: change the performance partway through a take.

WHY THIS WORKS AT ALL, AND WHERE IT HAS TO SIT
  `H3ChunkClose` expands the graph once per chunk, and the body it clones is
  everything downstream of `H3ChunkOpen` and upstream of Close. A node taking
  `chunk_index` is therefore INSIDE that body by construction, and gets cloned
  with a different index each time. So a model patch that reads the index is
  per-chunk for free -- no change to the chunking machinery.

  The corollary is that this node must sit BETWEEN the model loader and the
  guider, and it must take `chunk_index` from Open. Wire it anywhere that does
  not descend from Open and every chunk silently gets chunk 0's LoRA.

ADDRESSED BY TIME, NOT BY CHUNK NUMBER
  Same idiom as `H3 Shot List` and `H3 Dialogue`, and for the same reason:
  chunk boundaries move when you change the chunk size, and a schedule written
  against them breaks silently. Times survive a re-plan. Resolution goes
  through `chunk_windows`, which the dialogue node also uses, so a LoRA cue and
  a spoken line at 00:12 mean the same instant.

RAMP RATHER THAN SWAP, WHERE YOU CAN
  A LoRA swap at a chunk boundary is a discontinuity in the MODEL. The latent
  pin holds content across the join, but the weights generating the new chunk
  differ, so style and motion character can step exactly at the seam -- the one
  place a long take can least afford it. Ramping the strength of ONE LoRA has
  no such edge, and for "build the intensity" it is what you actually want.
  Write `0.4-0.9` as the strength to interpolate across the row's span.
"""

import logging

from .story import FPS, chunk_windows, parse_rows

CATEGORY = "MiniMax H3/long-form"


def parse_time(text):
    """'12', '12.5', '00:12', '01:02.5' -> seconds. None if unparsable."""
    s = str(text).strip()
    if not s:
        return None
    try:
        if ":" in s:
            mm, ss = s.rsplit(":", 1)
            return int(mm) * 60 + float(ss)
        return float(s)
    except ValueError:
        return None


def parse_span(text):
    """'00:00-00:12' -> (0.0, 12.0). A bare time means from there to the end."""
    s = str(text).strip()
    # rsplit on '-' would break '00:00-00:12' the same way either way, but a
    # leading minus is not a thing here, so split on the first separator that
    # is not inside a timestamp
    for sep in ("..", "-", " to "):
        if sep in s:
            a, b = s.split(sep, 1)
            lo, hi = parse_time(a), parse_time(b)
            if lo is not None and hi is not None:
                return lo, hi
    lo = parse_time(s)
    return (lo, float("inf")) if lo is not None else None


def parse_strength(text):
    """'0.8' -> (0.8, 0.8);  '0.4-0.9' -> (0.4, 0.9) for a ramp."""
    s = str(text).strip()
    for sep in ("..", "-"):
        if sep in s:
            a, b = s.split(sep, 1)
            try:
                return float(a), float(b)
            except ValueError:
                pass
    try:
        v = float(s)
    except ValueError:
        v = 1.0
    return v, v


def chunk_spans(chunks):
    """Each chunk's (start, end) on the FINISHED clip, in seconds.

    Not the chunk's own start frame: the join drops everything before
    `keep_from`, so what lands in the output is `keep_from`..`end`, placed after
    everything kept before it. `chunk_windows` already computes that offset for
    the dialogue node, and reusing it is what keeps the two schedules agreeing
    about when 00:12 is.
    """
    out = []
    for c, w in zip(chunks, chunk_windows(chunks)):
        head = (int(c["keep_from"]) - int(c["start"])) / FPS
        start = w["offset"] + head
        out.append((start, start + (int(c["end"]) - int(c["keep_from"])) / FPS))
    return out


def resolve(rows, span):
    """Rows overlapping this chunk's span, each with its strength here.

    A ramp is evaluated at the chunk's MIDPOINT. A chunk is the smallest unit
    that can carry a different patch, so a ramp is a staircase whatever we do;
    the midpoint makes each tread the average of the slope it covers rather
    than its leading edge.
    """
    lo, hi = span
    mid = (lo + hi) / 2.0
    hits = []
    for name, (a, b), (s0, s1) in rows:
        if b <= lo or a >= hi:
            continue
        if s0 == s1 or not (b > a) or b == float("inf"):
            strength = s0
        else:
            t = min(1.0, max(0.0, (mid - a) / (b - a)))
            strength = s0 + (s1 - s0) * t
        hits.append((name, round(strength, 4)))
    return hits


class H3ChunkLora:
    """Apply LoRAs to particular chunks of a chunked take, addressed by time."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            names = folder_paths.get_filename_list("loras")
        except Exception:  # pragma: no cover - outside ComfyUI
            names = []
        return {"required": {
            "model": ("MODEL", {"tooltip": "Put this BETWEEN the model loader "
                                           "and the guider."}),
            "chunk_index": ("INT", {"default": 0, "min": 0, "max": 4096,
                            "forceInput": True,
                            "tooltip": "From H3 Chunk Open. This is what puts "
                                       "the node inside the per-chunk body — "
                                       "without it every chunk gets chunk 0's "
                                       "LoRA and nothing says so."}),
            "schedule": ("STRING", {"multiline": True, "default":
                         "# time span | lora_1/2/3 | strength\n"
                         "# 00:00-00:20 | lora_1 | 0.4-0.9\n"
                         "# 00:20       | lora_2 | 0.8\n",
                         "tooltip": "Times are on the FINISHED clip. A strength "
                                    "range ramps across the span.\n\nName either "
                                    "a FILENAME, exactly as the pickers below "
                                    "spell it, or a picker slot (lora_1..3). "
                                    "Filenames have no ceiling — one node carries "
                                    "as many as you list — which is why H3 Script "
                                    "emits them and leaves the pickers empty. The "
                                    "slots are for schedules written by hand, "
                                    "where a dropdown beats typing a path."},),
        }, "optional": {
            "chunk_plan": ("H3_CHUNK_PLAN",),
            "lora_1": (names, {"tooltip": "Only needed if the schedule says "
                               "'lora_1'. A schedule naming files directly "
                               "leaves this empty."}),
            "lora_2": (names,),
            "lora_3": (names,),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Apply different LoRAs, or different strengths, to different "
                   "parts of a chunked take. Times are on the FINISHED clip.")

    _cache = {}

    @classmethod
    def _load(cls, name):
        import comfy.utils
        import folder_paths
        if name in cls._cache:
            return cls._cache[name]
        path = folder_paths.get_full_path("loras", name)
        if path is None:
            raise ValueError(f"H3 Chunk LoRA: '{name}' is not in the loras "
                             f"folder. Names come from the lora_1..3 pickers, "
                             f"or type the filename exactly.")
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        cls._cache[name] = sd
        return sd

    def go(self, model, chunk_index, schedule, chunk_plan=None,
           lora_1=None, lora_2=None, lora_3=None):
        # comfy.sd is imported where it is USED, below, so a bad schedule or a
        # missing chunk_plan reports itself instead of failing on an import
        rows = []
        for r in parse_rows(schedule):
            if len(r) < 2:
                continue
            span = parse_span(r[0])
            if span is None:
                logging.warning("H3ChunkLora: cannot read a time span from "
                                "%r — row skipped", r[0])
                continue
            rows.append((r[1], span,
                         parse_strength(r[2] if len(r) > 2 else "1.0")))
        slots = {"lora_1": lora_1, "lora_2": lora_2, "lora_3": lora_3}
        resolved = []
        for name, span, st in rows:
            key = name.strip().lower()
            if key in slots:
                got = slots[key]
                if not got or got in ("None", "(none)"):
                    raise ValueError(
                        f"H3 Chunk LoRA: the schedule names '{key}' but that "
                        f"picker is empty. Choose a file in the {key} widget, "
                        f"or write a filename in the schedule instead.")
                name = got
            resolved.append((name, span, st))
        rows = resolved

        if not rows:
            return {"ui": {"h3char": ["H3 CHUNK LORA — no rows, model passed "
                                      "through unchanged"]},
                    "result": (model, "no rows; model unchanged")}

        chunks = (chunk_plan or {}).get("chunks") if chunk_plan else None
        if not chunks:
            raise ValueError(
                "H3 Chunk LoRA: no chunk_plan wired. The schedule is in "
                "FINISHED-CLIP time and only the plan knows which frames a "
                "chunk contributes, so without it the rows cannot be placed.")

        i = max(0, min(int(chunk_index), len(chunks) - 1))
        spans = chunk_spans(chunks)
        hits = resolve(rows, spans[i])

        lines = [f"H3 CHUNK LORA — chunk {i + 1} of {len(chunks)}, "
                 f"{spans[i][0]:.2f}s to {spans[i][1]:.2f}s on the finished clip"]
        out = model
        for name, strength in hits:
            if abs(strength) < 1e-4:
                lines.append(f"  {name}: 0.0000, skipped")
                continue
            import comfy.sd
            out = comfy.sd.load_lora_for_models(out, None, self._load(name),
                                                strength, 0)[0]
            lines.append(f"  {name}: {strength:.4f}")
        if not hits:
            lines.append("  nothing scheduled here; model unchanged")

        # every chunk's assignment, so one look says whether the ramp is the
        # shape you meant -- a schedule that reads right and resolves wrong is
        # the failure this reports its way out of
        lines.append("  schedule across the take:")
        for k, sp in enumerate(spans):
            got = resolve(rows, sp)
            desc = ", ".join(f"{n.rsplit('.', 1)[0][-24:]} {s:.2f}"
                             for n, s in got) or "-"
            mark = " <-" if k == i else ""
            lines.append(f"    chunk {k + 1}: {sp[0]:6.2f}-{sp[1]:6.2f}s  "
                         f"{desc}{mark}")

        info = "\n".join(lines)
        logging.info("H3ChunkLora: chunk %d/%d -> %s", i + 1, len(chunks),
                     hits or "nothing")
        return {"ui": {"h3char": [info]}, "result": (out, info)}


NODE_CLASS_MAPPINGS = {"H3ChunkLora": H3ChunkLora}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkLora": "H3 Chunk LoRA (per-chunk)"}
