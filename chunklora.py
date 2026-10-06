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

ADDRESSED BY CHUNK, OR BY TIME
  BY CHUNK is usually what you mean: `chunk 2`, `chunk 2-4`, `chunk 2,5`,
  `chunk 2+`, `chunk last`. A chunk reference stays a set of INDICES and matches
  by identity, so it cannot bleed into a neighbour.

  BY TIME -- `00:14-00:19` -- is still here and still right for a cue that has
  to line up with something else on the clock: resolution goes through
  `chunk_windows`, which H3 Dialogue also uses, so a LoRA cue and a spoken line
  at 00:12 mean the same instant.

  THE TRADE, measured on a 4x141/39 plan where chunk 3 runs to 14.38s:

      00:14-00:19 | lora    applies to chunks 3 AND 4
      chunk last  | lora    applies to chunk 4

  A time span is matched by OVERLAP, and a boundary is a shared edge, so "from
  14 seconds" catches the chunk that is still running at 14 seconds. Worse,
  every boundary moves when chunk_frames or chunk_count changes, and the whole
  rest of this pack is addressed in frames and chunks. The time form was the
  only form for a while and it made "put this on the last chunk" -- which has no
  stable timestamp at all -- the hardest thing to write.

  Chunk references CAN break on a re-plan: that is the real cost, and the
  objection the time form was built to dodge. So a row naming a chunk the plan
  does not have RAISES, naming both numbers. Silently applying nothing is how a
  take comes back without its LoRA and nobody knows why.

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


def parse_chunks(text):
    """`chunk 2` / `2-4` / `2,5` / `2+` / `last` -> a spec, or None.

    WHY THIS IS NOT CONVERTED TO A TIME SPAN
      Resolving `chunk 2` to chunk 2's seconds and then matching by overlap puts
      the bleed straight back: a boundary is a shared edge, and a span that
      touches it applies to BOTH neighbours. That is the whole complaint. A
      chunk reference stays a set of INDICES and matches by identity, so a row
      naming chunk 2 cannot reach chunk 3 wherever the boundary lands.

    Indices are 1-BASED here, matching the `chunk N` the report prints.
    """
    s = str(text).strip().lower()
    if not s.startswith("chunk"):
        return None
    body = s[len("chunk"):].strip()
    if not body:
        return None
    parts = [b.strip() for b in body.split(",") if b.strip()]
    spec = []
    for b in parts:
        if b in ("last", "end", "-1"):
            spec.append("last")
        elif b.endswith("+"):
            head = b[:-1].strip()
            if not head.isdigit():
                return None
            spec.append(("from", int(head)))
        elif "-" in b or ".." in b:
            sep = ".." if ".." in b else "-"
            a, c = (x.strip() for x in b.split(sep, 1))
            if not (a.isdigit() and c.isdigit()):
                return None
            spec.append(("range", int(a), int(c)))
        elif b.isdigit():
            spec.append(int(b))
        else:
            return None
    return spec or None


def chunk_indices(spec, total):
    """A spec plus the plan's chunk count -> (0-based set, out-of-range names)."""
    out, bad = set(), []
    for item in spec:
        if item == "last":
            if total:
                out.add(total - 1)
        elif isinstance(item, int):
            if 1 <= item <= total:
                out.add(item - 1)
            else:
                bad.append(str(item))
        elif item[0] == "from":
            a = item[1]
            if a > total:
                bad.append(f"{a}+")
            out.update(range(max(0, a - 1), total))
        elif item[0] == "range":
            a, b = item[1], item[2]
            if a > total or b > total:
                bad.append(f"{a}-{b}")
            out.update(range(max(0, a - 1), min(total, b)))
    return out, bad


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


def resolve(rows, span, index=None, total=None):
    """Rows that apply to this chunk, each with its strength here.

    A row is either TIME-addressed, matched by overlapping `span`, or
    CHUNK-addressed, matched by containing `index`. A time ramp is evaluated at
    the chunk's MIDPOINT: a chunk is the smallest unit that can carry a
    different patch, so a ramp is a staircase whatever we do, and the midpoint
    makes each tread the average of the slope it covers rather than its leading
    edge. A chunk ramp interpolates across the chunks the row names, by their
    position in that set -- so `chunk 2-4 | 0.4-0.9` is 0.4, 0.65, 0.9 and not
    three readings off a clock.
    """
    lo, hi = span
    mid = (lo + hi) / 2.0
    hits = []
    for row in rows:
        name, target, (s0, s1) = row
        if isinstance(target, tuple) and target and target[0] == "chunks":
            members = sorted(target[1])
            if index is None or index not in members:
                continue
            if s0 == s1 or len(members) < 2:
                strength = s0
            else:
                t = members.index(index) / (len(members) - 1)
                strength = s0 + (s1 - s0) * t
        else:
            a, b = target
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
            # "(none)" FIRST, and the default. A COMBO has no empty state: the
            # frontend always sends one of the options and the server validates
            # against the list, so without a sentinel there is NO legal way to
            # leave a picker unused -- you had to park an arbitrary filename in
            # it, which then looks armed and is not. H3 38 shipped with two junk
            # entries for exactly that reason. The resolver already tested for
            # "(none)"; it was only ever missing from the list.
            names = ["(none)"] + folder_paths.get_filename_list("loras")
        except Exception:  # pragma: no cover - outside ComfyUI
            names = ["(none)"]
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
                         "# where | lora_1/2/3 or a filename | strength\n"
                         "# chunk 2-4   | lora_1 | 0.4-0.9\n"
                         "# chunk last  | lora_2 | 0.8\n"
                         "# 00:20       | lora_3 | 0.8\n",
                         "tooltip": "One row per cue. The first field says WHERE, "
                                    "by chunk or by time.\n\n"
                                    "BY CHUNK, 1-based, and exact — it cannot "
                                    "bleed into a neighbour:\n"
                                    "  chunk 2      that chunk only\n"
                                    "  chunk 2-4    those three, ramping\n"
                                    "  chunk 2,5    two specific chunks\n"
                                    "  chunk 2+     chunk 2 to the end\n"
                                    "  chunk last   the final chunk\n\n"
                                    "BY TIME, on the FINISHED clip — for a cue "
                                    "that must match a dialogue line. Matched by "
                                    "OVERLAP, so a span touching a boundary "
                                    "applies to BOTH chunks: on a 4x141/39 plan "
                                    "chunk 3 runs to 14.38s, and 00:14-00:19 hits "
                                    "chunks 3 and 4. Prefer `chunk last`.\n\n"
                                    "A strength RANGE (0.4-0.9) ramps — across "
                                    "the named chunks, or across a time span by "
                                    "each chunk's midpoint.\n\n"
                                    "Name a FILENAME exactly as the pickers spell "
                                    "it, or a picker slot (lora_1..3). Filenames "
                                    "have no ceiling — one node carries as many as "
                                    "you list — which is why H3 Script emits them "
                                    "and leaves the pickers on (none). Several "
                                    "rows may name the same file, and several may "
                                    "target one chunk; they stack, and nothing "
                                    "dedupes, so the same file twice over one "
                                    "chunk applies twice."},),
        }, "optional": {
            "chunk_plan": ("H3_CHUNK_PLAN", {"tooltip":
                "The plan, so a schedule written in FINISHED-CLIP time can be "
                "converted to this chunk's own frames. Without it the node has no "
                "way to know where chunk 3 sits in the take, and a span like "
                "00:15-00:19 cannot be resolved at all.\n\n"
                "A span is included when it OVERLAPS the chunk at any point, not "
                "when its midpoint falls inside — so a span crossing a chunk "
                "boundary applies to BOTH chunks. Check the info output against "
                "the plan's chunk times before trusting a tight span."}),
            "lora_1": (names, {"default": "(none)",
                               "tooltip": "Only needed if the schedule says "
                               "'lora_1'. Leave it on (none) when the schedule "
                               "names files directly — a picker is read ONLY when "
                               "a row names its slot, and nothing is loaded for a "
                               "chunk with no matching row."}),
            "lora_2": (names, {"default": "(none)",
                               "tooltip": "Only needed if the schedule says "
                               "'lora_2'. A schedule naming files directly leaves "
                               "this empty."}),
            "lora_3": (names, {"default": "(none)",
                               "tooltip": "Only needed if the schedule says "
                               "'lora_3'. Three pickers is not a limit on how "
                               "many LoRAs a schedule can drive — naming "
                               "filenames instead has no ceiling."}),
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
        # AN EMPTY SCHEDULE COSTS NOTHING AND NEEDS NOTHING. Checked before the
        # plan, because a bypassed-in-effect node should not demand wiring to do
        # nothing -- I moved the plan check up here first and broke exactly that.
        raw = [r for r in parse_rows(schedule) if len(r) >= 2]
        if not raw:
            return {"ui": {"h3char": ["H3 CHUNK LORA — no rows, model passed "
                                      "through unchanged"]},
                    "result": (model, "no rows; model unchanged")}

        # The plan is needed BEFORE the rows are resolved, because `chunk last`
        # and `chunk 2+` cannot be placed without knowing how many chunks exist.
        chunks = (chunk_plan or {}).get("chunks") if chunk_plan else None
        if not chunks:
            raise ValueError(
                "H3 Chunk LoRA: no chunk_plan wired. A schedule addressed by "
                "CHUNK needs the count to resolve `last` and `N+`, and one "
                "addressed by TIME needs to know which frames a chunk "
                "contributes. Wire H3 Chunk Plan's `plan`.")
        total = len(chunks)

        rows, bad_chunks = [], []
        for r in raw:
            spec = parse_chunks(r[0])
            if spec is not None:
                members, out_of_range = chunk_indices(spec, total)
                if out_of_range:
                    # LOUDLY. A row naming a chunk the plan does not have is the
                    # re-plan hazard that addressing by time was meant to avoid;
                    # silently applying nothing is how a take comes back without
                    # the LoRA and nobody knows why.
                    bad_chunks.append((r[0].strip(), out_of_range))
                if not members:
                    continue
                target = ("chunks", members)
            else:
                target = parse_span(r[0])
                if target is None:
                    logging.warning("H3ChunkLora: cannot read a time span or a "
                                    "chunk reference from %r — row skipped", r[0])
                    continue
            rows.append((r[1], target,
                         parse_strength(r[2] if len(r) > 2 else "1.0")))
        if bad_chunks:
            detail = "; ".join(f"{src!r} names chunk(s) {', '.join(oor)}"
                               for src, oor in bad_chunks)
            raise ValueError(
                f"H3 Chunk LoRA: the plan has {total} chunk(s), but {detail}. "
                f"Either the plan changed under the schedule or the schedule is "
                f"wrong — both are worth stopping for, because the alternative "
                f"is a take that renders without the LoRA and looks fine.")
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

        i = max(0, min(int(chunk_index), total - 1))
        spans = chunk_spans(chunks)
        hits = resolve(rows, spans[i], i, total)

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
            got = resolve(rows, sp, k, total)
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
