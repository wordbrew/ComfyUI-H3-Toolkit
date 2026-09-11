"""One document instead of five nodes that have to agree.

WHAT THIS IS FOR
  H3's prompt format is index-bookkeeping: `<Subject 1>`, `<Picture 3>`,
  `<Audio 1>`, and a retention_analysis block that has to name the same subjects
  the definitions do. That bookkeeping currently lives in a human's head, spread
  across H3 Character, H3 Scene Prompt, H3 Shot List, H3 Dialogue and H3
  Long-Form Links -- five nodes whose fields must agree and which nothing checks
  against each other.

  On 2026-09-02 they did not agree, and a take rendered a reference image
  instead of the shot because subject_definitions said "preserve her exactly as
  shown in those pictures" while the description asked for motion. Nothing
  errored. That is the failure this removes: you write `@ada` and the compiler
  decides what number she is.

A DOCUMENT, NOT A TEXT FORMAT
  `parse()` produces a plain dict and `emit()` renders H3's fields from it. The
  DSL is one serialization of that dict, not the thing itself. That is
  deliberate and it is the whole reason a timeline editor is possible later: a
  UI edits the document and re-serializes, and text and timeline stay in sync
  because neither is authoritative. Writing this as text-in/strings-out would
  have made the app a rewrite rather than a view.

WHAT IT DOES NOT DO, ON PURPOSE
  It does not place dialogue in time. `H3Dialogue` already does that, including
  the rule that matters -- no line may start inside a chunk's pin, because those
  frames are stamped back from the previous chunk at every denoise step and the
  speech is simply never said. One implementation of that, not two. This emits
  lines in H3Dialogue's own input format and lets it do the timing.

  Anchor counts come FROM the character store, not from you. An earlier cut
  made `pictures` a required declaration on the grounds that a folder changing
  underneath you should not silently renumber `<Picture n>`. That protected the
  compiler, not the author: nobody wants to type a number they already
  established when they saved the character. The numbering is printed in `info`
  every run, so a change is visible rather than silent.

GRAMMAR
    # comment
    @name  = <description>              a subject
    @name  = character <StoreName>      a subject whose anchors are in the store
    @name  = setting. <description>     the setting
  Five keywords and `@name`. That is the whole surface a take needs:

    shot | ...      note ...      say @name ...      do ...      ---

  A character's pictures, voice, description and retention all come from the
  store. Everything below is an OVERRIDE, rarely typed, and exists mainly so a
  form-shaped UI has fields to bind to -- the document carries more than the
  text syntax asks for:

    @name.retention = fully_preserved | partially_preserved | free
    @name.preserve  = facial identity, hair, eye colour and build
    @name.allow     = natural movement and changing expression
    @name.retention_detail = <replaces the whole clause>
    @name.pictures  = 3        only when there is no store entry
    @name.audio     = 1

  RETENTION IS ABOUT LIKENESS, and only that. `preserve` and `allow` scope what
  stays the same about a subject between shots. Pose, motion and performance go
  in the shot -- `note` and `do` -- not here.

    task       = reference generation | video editing
    soundscape = ...
    music      = ...

    shot | <description>
      note <text>                       adds to the description
      say  @name [verb] <line>          dialogue, in order
      say  @name | how they say it | <line>
                                        the same, when "how" is a PHRASE --
                                        "says quietly, half-turning away". The
                                        six bare verbs are a shortcut, not the
                                        vocabulary; H3Dialogue takes any phrase.
      do   <text>                       action for the current chunk
      lora <file> <strength|a -> b>
      ---                               chunk break, as in H3 Dialogue
"""

import json
import re

CATEGORY = "MiniMax H3/prompt"

RETENTIONS = ("fully_preserved", "partially_preserved", "free")
# the verbs that need no pipe. NOT the vocabulary -- any phrase works via
# `say @name | how they say it | the line`, which is what H3Dialogue takes.
SPEECH_VERBS = ("moans", "asks", "whispers", "says", "shouts", "sighs")
NAME = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


def store_card(name):
    """(anchor count, voice count, description, retention) from the store.

    Reads the same `models/h3_characters/<name>/` layout H3 Character writes,
    so a character saved once is usable by name here with nothing re-typed.
    Missing store, missing character or no ComfyUI at all -> all zeros, and the
    script's own overrides take over.
    """
    try:
        import os

        from .character import characters_dir
        d = os.path.join(characters_dir(), name)
        if not os.path.isdir(d):
            return 0, 0, "", ""
        img = os.path.join(d, "images")
        n = len([f for f in sorted(os.listdir(img))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                ) if os.path.isdir(img) else 0
        voice = 1 if os.path.exists(os.path.join(d, "voice.wav")) else 0
        desc, ret = "", ""
        card = os.path.join(d, "card.json")
        if os.path.exists(card):
            with open(card, encoding="utf-8") as fh:
                data = json.load(fh)
            desc = (data.get("description") or "").strip()
            ret = (data.get("retention") or "").strip()
        return n, voice, desc, ret
    except Exception:
        return 0, 0, "", ""


class ScriptError(ValueError):
    """A problem with the script, reported with the line it is on."""


def _err(lineno, text, line):
    return ScriptError(f"line {lineno}: {text}\n    {line.strip()}")


def parse(text):
    """The DSL -> the document. Pure data; no ComfyUI, no torch."""
    doc = {"version": 1, "cast": [], "shots": [],
           "task": "reference generation", "soundscape": "", "music": "N/A"}
    by_name = {}
    shot = None

    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.rstrip()
        body = line.strip()
        if not body or body.startswith("#"):
            continue

        # --- cast and settings, at the left margin ------------------------- #
        if not line[:1].isspace():
            if body.startswith("@"):
                head, sep, rest = body.partition("=")
                if not sep:
                    raise _err(lineno, "a cast line needs '='", line)
                head, rest = head.strip(), rest.strip()
                if "." in head:
                    nm, _, attr = head[1:].partition(".")
                    entry = by_name.get(nm)
                    if entry is None:
                        raise _err(lineno, f"@{nm} is not declared yet", line)
                    if attr == "retention":
                        if rest not in RETENTIONS:
                            raise _err(lineno, f"retention must be one of "
                                       f"{', '.join(RETENTIONS)}", line)
                        entry["retention"] = rest
                    elif attr in ("pictures", "audio"):
                        try:
                            entry[attr] = int(rest)
                        except ValueError:
                            raise _err(lineno, f"{attr} takes a number", line)
                    elif attr in ("preserve", "allow", "retention_detail"):
                        # `preserve` lists what stays the same, `allow` what may
                        # vary -- both LIKENESS scoping. `retention_detail`
                        # replaces the whole clause when the default wording is
                        # not what a shot needs.
                        entry[attr] = rest
                    else:
                        raise _err(lineno, f"unknown attribute '{attr}'", line)
                    continue
                nm = head[1:]
                if nm in by_name:
                    raise _err(lineno, f"@{nm} is declared twice", line)
                entry = {"name": nm, "kind": "subject", "description": rest,
                         "retention": "partially_preserved", "pictures": 0,
                         "audio": 0}
                if rest.startswith("character "):
                    entry["kind"] = "character"
                    entry["character"] = rest[len("character "):].strip()
                    entry["description"] = ""
                    entry["retention"] = "fully_preserved"
                    # from the store, so nothing already established when the
                    # character was saved has to be typed again here
                    npic, naud, desc, ret = store_card(entry["character"])
                    entry["pictures"] = npic
                    entry["audio"] = naud
                    if desc:
                        entry["store_description"] = desc
                    if ret in RETENTIONS:
                        entry["retention"] = ret
                    entry["from_store"] = bool(npic or desc)
                elif rest.startswith("setting."):
                    entry["kind"] = "setting"
                    entry["description"] = rest[len("setting."):].strip()
                doc["cast"].append(entry)
                by_name[nm] = entry
                continue

            key, sep, rest = body.partition("=")
            key = key.strip()
            if sep and key in ("task", "soundscape", "music"):
                doc[key] = rest.strip()
                continue
            if body.startswith("shot"):
                head, _, desc = body.partition("|")
                # A SHOT HAS A LENGTH, because a shot is a cut and a cut is where
                # the planner opens a chunk. Optional: a shot with no number is
                # sized by the planner as before, which keeps every script
                # written before this parseable.
                frames = None
                rest_head = head[len("shot"):].strip()
                if rest_head:
                    try:
                        frames = int(rest_head)
                    except ValueError:
                        raise _err(lineno, f"a shot's length must be a frame "
                                           f"count, got {rest_head!r}", line)
                    if frames < 5:
                        raise _err(lineno, "the shortest legal run is 5 frames",
                                   line)
                shot = {"description": desc.strip(), "notes": [],
                        "chunks": [{"lines": [], "actions": []}], "loras": []}
                if frames is not None:
                    shot["frames"] = frames
                doc["shots"].append(shot)
                continue
            raise _err(lineno, "expected a cast line, a setting, or 'shot |'",
                       line)

        # --- inside a shot -------------------------------------------------- #
        if shot is None:
            raise _err(lineno, "indented line before any 'shot |'", line)
        verb, _, rest = body.partition(" ")
        rest = rest.strip()
        cur = shot["chunks"][-1]

        if body.startswith("---"):
            shot["chunks"].append({"lines": [], "actions": []})
        elif verb == "note":
            shot["notes"].append(rest)
        elif verb == "do":
            cur["actions"].append(rest)
        elif verb == "say":
            m = NAME.match(rest)
            if not m:
                raise _err(lineno, "say needs a @name", line)
            who = m.group(1)
            if who not in by_name:
                raise _err(lineno, f"@{who} is not declared", line)
            after = rest[m.end():].strip()
            # PLACED BY HAND: `@2.5` is seconds from the shot's own start. A cast
            # name can never begin with a digit (see NAME), so this cannot collide
            # with one. Without it a line simply flows after the one before, which
            # is what H3Dialogue does on its own.
            at = None
            if after.startswith("@"):
                tok, _, tail = after.partition(" ")
                try:
                    at = float(tok[1:])
                except ValueError:
                    raise _err(lineno, f"expected a time like @2.5 after the "
                                       f"speaker, got {tok!r}", line)
                if at < 0:
                    raise _err(lineno, "a placed time cannot be negative", line)
                after = tail.strip()
            # HOW IT IS SAID IS FREE TEXT, and that is where the prompting room is
            # -- "says quietly, half-turning away" reads nothing like "says". Six
            # bare words used to be the entire vocabulary here, which threw the
            # rest of the phrase into the spoken line. H3Dialogue downstream has
            # always taken a free verb (its own default is "says lightly"), so the
            # limit was ours alone.
            #
            # The pipe is the escape, and it is the separator H3Dialogue already
            # uses:  say @ada | says quietly, half-turning | Come here.
            # A bare recognised verb still works, because most lines want one.
            speech_verb = "says"
            if after.startswith("|"):
                parts = [p.strip() for p in after[1:].split("|", 1)]
                if len(parts) != 2:
                    raise _err(lineno, "a piped say needs '| how they say it | "
                                       "the line'", line)
                speech_verb, after = parts[0] or "says", parts[1]
            else:
                for v in SPEECH_VERBS:
                    if after.startswith(v + " "):
                        speech_verb, after = v, after[len(v):].strip()
                        break
            if not after:
                raise _err(lineno, "say has no line to speak", line)
            entry_line = {"who": who, "verb": speech_verb, "line": after}
            if at is not None:
                entry_line["at"] = at
            cur["lines"].append(entry_line)
        elif verb == "lora":
            parts = rest.replace("->", " ").split()
            if not parts:
                raise _err(lineno, "lora needs a file name", line)
            name = parts[0]
            nums = [p for p in parts[1:] if p]
            strength = "-".join(nums) if len(nums) > 1 else (
                nums[0] if nums else "1.0")
            shot["loras"].append({"name": name, "strength": strength})
        else:
            raise _err(lineno, f"unknown keyword '{verb}'", line)

    if not doc["cast"]:
        raise ScriptError("no cast: declare at least one @name")
    return doc


_SETTING_DEFAULTS = {"task": "reference generation", "soundscape": "", "music": "N/A"}
# order matters: parse() reads these off `@name.<attr> = ...` lines
_CAST_ATTRS = ("retention", "pictures", "audio", "preserve", "allow",
               "retention_detail")


def _head_line(entry):
    """The `@name = ...` line for one cast entry."""
    if entry.get("kind") == "character":
        return f"@{entry['name']} = character {entry.get('character', '')}".rstrip()
    if entry.get("kind") == "setting":
        return f"@{entry['name']} = setting. {entry.get('description', '')}".rstrip()
    return f"@{entry['name']} = {entry.get('description', '')}".rstrip()


def serialize(doc):
    """The document -> the DSL. The inverse of `parse`, and the half a UI needs.

    WHY THIS EXISTS
      `parse` turned text into a document and `emit` turned a document into H3's
      fields, but nothing turned a document back into text. That is the direction
      an editor writes in: the panel edits the document, serializes, and the text
      in the node's widget stays the readable form of the same thing. Without it
      a UI has to own its own storage and the two drift.

    WHAT ROUND-TRIPPING MEANS HERE
      `parse(serialize(doc)) == doc`, NOT `serialize(parse(text)) == text`.
      Comments, blank lines and the author's spacing are not in the document and
      cannot come back. The document is what has to survive.

    OVERRIDES ARE EMITTED ONLY WHEN THEY DIFFER
      A character's pictures, audio, description and retention come from the
      store, so writing them back would freeze today's values into the script and
      a re-saved character would stop taking effect. Rather than special-case
      that, each head line is re-parsed on its own and only the attributes that
      differ from that baseline are written -- which consults the store exactly
      the way a later `parse` will.
    """
    doc = doc or {}
    out = []

    for entry in doc.get("cast", []):
        head = _head_line(entry)
        out.append(head)
        try:
            base = parse(head)["cast"][0]
        except ScriptError:
            base = {}
        for attr in _CAST_ATTRS:
            if attr not in entry:
                continue
            if base.get(attr) == entry[attr]:
                continue
            if entry[attr] in ("", None):
                continue
            out.append(f"@{entry['name']}.{attr} = {entry[attr]}")

    settings = [f"{k} = {doc[k]}" for k, default in _SETTING_DEFAULTS.items()
                if doc.get(k, default) != default]
    if settings:
        out.append("")
        out.extend(settings)

    for shot in doc.get("shots", []):
        out.append("")
        frames = shot.get("frames")
        head = f"shot {int(frames)} |" if frames else "shot |"
        out.append(f"{head} {shot.get('description', '')}".rstrip())
        for note in shot.get("notes", []):
            out.append(f"  note {note}")
        for lora in shot.get("loras", []):
            strength = str(lora.get("strength", "1.0")).replace("-", " ")
            out.append(f"  lora {lora['name']} {strength}".rstrip())
        for i, chunk in enumerate(shot.get("chunks", [])):
            if i:
                out.append("  ---")
            for action in chunk.get("actions", []):
                out.append(f"  do   {action}")
            for line in chunk.get("lines", []):
                verb = (line.get("verb") or "says").strip()
                at = line.get("at")
                where = f"@{line['who']}" + (f" @{float(at):g}" if at is not None else "")
                if verb in SPEECH_VERBS and "|" not in line["line"]:
                    out.append(f"  say  {where} {verb} {line['line']}")
                else:
                    # a phrase, or a line containing a pipe: the explicit form
                    out.append(f"  say  {where} | {verb} | {line['line']}")

    return "\n".join(out).strip() + "\n"


def cuts_from(doc, chunk_frames=141):
    """Shot lengths -> the cut frames H3 Chunk Plan takes.

    A shot IS a cut: the planner opens a chunk at each one, and that chunk gets
    no carried prefix, which is where the contrast carry resets. Shots with no
    length of their own are sized at `chunk_frames`, so a script written before
    lengths existed still plans the way it always did.
    """
    cuts, at = [], 0
    for shot in doc.get("shots", []):
        if at:
            cuts.append(at)
        at += int(shot.get("frames") or chunk_frames)
    return cuts, at


def build_plan(doc, chunk_frames=141, context=39, mode="fixed",
               total_frames=None):
    """The chunk plan a take implies -> (chunks, info), H3 Chunk Plan's own shape.

    WHY THIS LIVES HERE
      The board draws the plan the shot lengths imply. The render used whatever
      H3 Chunk Plan had in its widgets. Nothing checked that those agreed, so the
      timeline could show one thing and ComfyUI render another -- a tool that
      lies is worse than no tool.

      Emitting the plan makes the two the same object rather than two things that
      ought to match. It calls `chunkplan.plan`, the same function H3 Chunk Plan
      calls, so this is not a second planner.

      H3 Chunk Plan stays: plenty of takes have no script.
    """
    from .chunkplan import plan as _plan
    cuts, from_shots = cuts_from(doc, chunk_frames)
    total = int(total_frames or from_shots or chunk_frames)
    return _plan(total, chunk_frames=int(chunk_frames), mode=mode,
                 cuts=cuts or None, context=int(context))


def audio_clock_notes(doc, chunk_frames=141):
    """Shots whose length drifts the two clocks apart.

    24 fps video and 40 Hz audio only line up when the frame count divides by 3.
    Off-clock costs SYNC DRIFT that compounds down a chain -- not artifacts, which
    is why it is easy to miss. 39, 90, 141, 192, 243, 294, 345 and 396 sit on both.
    """
    notes = []
    for i, shot in enumerate(doc.get("shots", [])):
        n = int(shot.get("frames") or chunk_frames)
        if n % 17 != 5:
            notes.append(f"shot {i + 1} is {n} frames, not a legal run (17n+5) — "
                         f"the planner will trim it, so the length you designed "
                         f"to is not the length rendered")
        elif n % 3:
            notes.append(f"shot {i + 1} is {n} frames, which is not on the audio "
                         f"clock — the 24 fps and 40 Hz grids drift apart. It "
                         f"costs sync, not artifacts, so it is easy to miss.")
    return notes


def lora_schedule(doc, chunks):
    """Per-shot `lora` lines -> H3 Chunk Lora's schedule, plus the slot map.

    TWO THINGS THE COMPILER IS FOR, again. H3 Chunk Lora addresses LoRAs by SLOT
    (`lora_1`..`lora_3`), not by filename, and by time on the FINISHED clip --
    which is not the time on the source, because the join drops each chunk's
    carried handle. Writing that schedule by hand means holding both conversions
    in your head while editing shots that move.

    A shot's finished start is the kept frames before it. `chunk_windows` derives
    the same number for dialogue; here it is simply the running total of
    `end - keep_from`.

    -> (schedule text, {slot: filename}, notes)
    """
    slots, order = {}, []
    for shot in doc.get("shots", []):
        for lora in shot.get("loras", []):
            name = lora.get("name")
            if name and name not in order:
                order.append(name)
    notes = []
    for i, name in enumerate(order[:3]):
        slots[f"lora_{i + 1}"] = name
    if len(order) > 3:
        notes.append(f"{len(order)} different LoRAs are named but H3 Chunk Lora "
                     f"has three slots — {', '.join(order[3:])} will not be "
                     f"scheduled")
    by_name = {n: f"lora_{i + 1}" for i, n in enumerate(order[:3])}

    # where each shot sits on the FINISHED clip
    finished, pos = [], 0
    for c in chunks:
        kept = int(c["end"]) - int(c["keep_from"])
        finished.append((int(c["keep_from"]), pos, kept))
        pos += kept

    def stamp_at(frames):
        s = frames / 24.0
        return f"{int(s // 60):02d}:{int(round(s % 60)):02d}"

    rows, at = [], 0
    for shot in doc.get("shots", []):
        n = int(shot.get("frames") or 0)
        span = [f for f in finished if at <= f[0] < at + (n or 10 ** 9)]
        if shot.get("loras") and span:
            start = span[0][1]
            end = span[-1][1] + span[-1][2]
            for lora in shot["loras"]:
                slot = by_name.get(lora.get("name"))
                if not slot:
                    continue
                rows.append(f"{stamp_at(start)}-{stamp_at(end)} | {slot} | "
                            f"{lora.get('strength', '1.0')}")
        at += n or 0
    return "\n".join(rows), slots, notes


def timing(doc, total_frames=None, chunk_frames=141, context=39, mode="fixed",
           syllables_per_second=4.3, gap=0.4, tail_margin=0.6, lead_in=0.075):
    """Where every line lands, which chunk holds it, and what will be lost.

    THE FAILURE THIS MAKES VISIBLE
      The first `pin` frames of every chunk after the first are reproduced from
      the previous chunk under a denoise mask of 0. Nothing new happens there, so
      a line starting inside that region is NEVER SAID -- measured 2026-08-28,
      and invisible until you listen to a finished render. A line running past
      its chunk's end is cut off mid-word instead.

      Auto-flowed lines cannot land in a handle, because they start at the chunk
      floor. A line PLACED by hand can, and that is the one way to lose speech
      silently -- so placed lines are checked against the floor and flowed ones
      are not, rather than warning about something that cannot happen.

    It reuses the planner and the window arithmetic H3Dialogue uses
    (`chunkplan.plan`, `story.chunk_windows`), so this is what will happen rather
    than a second opinion about it.
    """
    from .story import chunk_windows, syllables

    cuts, from_shots = cuts_from(doc, chunk_frames)
    total = int(total_frames or from_shots or chunk_frames)
    chunks, _info = build_plan(doc, chunk_frames=chunk_frames, context=context,
                               mode=mode, total_frames=total)
    windows = chunk_windows(chunks, lead_in=lead_in, tail_margin=tail_margin)

    # which chunks belong to which shot, by frame range.
    #
    # ONLY when lengths were actually declared. A script that never gave a shot a
    # length used to let its dialogue spread over the whole take, and capping it
    # at chunk_frames would silently shorten every take written before lengths
    # existed. No lengths -> no bounds -> the old behaviour exactly.
    sized = any(s.get("frames") for s in doc.get("shots", []))
    bounds, at = [], 0
    if sized:
        for shot in doc.get("shots", []):
            n = int(shot.get("frames") or chunk_frames)
            bounds.append((at, at + n))
            at += n

    # A CHUNK'S KEPT REGION IS keep_from..end, NOT start..end. Chunks overlap by
    # the carried handle -- chunk 1 of a 141-frame shot starts at 102, 39 frames
    # back inside chunk 0 -- so mapping by `start` files a chunk under the
    # previous shot and puts its lines in the wrong place.
    def kept(i):
        c = chunks[i]
        return int(c["keep_from"]), int(c["end"])

    def shot_windows(si):
        if si >= len(bounds):
            return list(enumerate(windows))
        lo, hi = bounds[si]
        out = [(i, w) for i, w in enumerate(windows)
               if kept(i)[0] < hi and kept(i)[1] > lo]
        return out or [(min(range(len(windows)),
                            key=lambda i: abs(kept(i)[0] - lo)), windows[0])][:1]

    out_lines, problems = [], []
    for si, shot in enumerate(doc.get("shots", [])):
        mine = shot_windows(si)
        if not mine:
            continue
        base = bounds[si][0] if si < len(bounds) else 0
        wi, cursor = 0, None
        for ch in shot.get("chunks", []):
            for line in ch.get("lines", []):
                text = line.get("line", "")
                dur = syllables(text) / max(0.1, float(syllables_per_second))
                rec = {"shot": si, "who": line.get("who", ""), "line": text,
                       "seconds": round(dur, 2), "placed": "at" in line,
                       "problem": None, "chunk": None, "start": None}

                if "at" in line:
                    # seconds from the SHOT's start; find the chunk covering it
                    abs_f = base + float(line["at"]) * 24.0
                    hit = next((iw for iw in mine
                                if kept(iw[0])[0] <= abs_f < kept(iw[0])[1]),
                               mine[-1])
                    ci, w = hit
                    local = abs_f / 24.0 - chunks[ci]["start"] / 24.0
                    if local < w["lo"] - 1e-6:
                        rec["problem"] = (
                            f"starts inside the {w['pin_s']:.2f}s carried handle "
                            f"— those frames are stamped back from the chunk "
                            f"before, so it is never said")
                    elif local + dur > w["hi"]:
                        rec["problem"] = "runs past the end of its chunk — cut off mid-word"
                    rec["chunk"] = ci
                    rec["start"] = round(w["offset"] + local, 2)
                    rec["local"] = round(local, 2)
                else:
                    if wi >= len(mine):
                        rec["problem"] = "no room left in this shot"
                    else:
                        ci, w = mine[wi]
                        if cursor is None:
                            cursor = w["lo"]
                        if cursor + dur > w["hi"]:
                            wi += 1
                            if wi < len(mine):
                                ci, w = mine[wi]
                                cursor = w["lo"]
                                if cursor + dur > w["hi"]:
                                    rec["problem"] = ("longer than a whole chunk "
                                                      "— it will be cut off")
                            else:
                                rec["problem"] = "no room left in this shot"
                        if rec["problem"] != "no room left in this shot":
                            rec["chunk"] = ci
                            rec["start"] = round(w["offset"] + cursor, 2)
                            rec["local"] = round(cursor, 2)
                            cursor = cursor + dur + float(gap)
                if rec["problem"]:
                    problems.append(f"{rec['problem']}: \u201c{text}\u201d")
                out_lines.append(rec)

    problems.extend(audio_clock_notes(doc, chunk_frames))

    lost = sum(w["pin_s"] for w in windows)
    speech = sum(max(0.0, w["hi"] - w["lo"]) for w in windows)
    if lost > 0:
        problems.append(
            f"the carried handles cost {lost:.1f}s of the take: every chunk after "
            f"a cut reproduces its opening from the one before, so nothing can be "
            f"said there. {speech:.1f}s is sayable in total.")

    return {
        "total_frames": total,
        "cuts": cuts,
        "chunks": [{"index": i, "shot": next((s for s, (lo, hi) in enumerate(bounds)
                                              if lo <= c["keep_from"] < hi), None),
                    "start": c["keep_from"], "frames": c["end"] - c["keep_from"],
                    "source_start": c["start"], "run": c["run"],
                    "pin": c.get("pin", 0),
                    "pin_s": round(w["pin_s"], 2), "run_s": round(w["run_s"], 2),
                    "speech_from": round(w["lo"], 2), "speech_to": round(w["hi"], 2),
                    "starts_at": round(w["offset"] + w["lo"], 2)}
                   for i, (c, w) in enumerate(zip(chunks, windows))],
        "shots": [{"index": i, "start": lo, "frames": hi - lo}
                  for i, (lo, hi) in enumerate(bounds)],
        "lines": out_lines,
        "problems": problems,
        "sayable_seconds": round(speech, 2),
        "pinned_seconds": round(lost, 2),
    }


def index(doc):
    """Assign <Subject n>, <Picture n> and <Audio n>. Declaration order wins.

    The one job worth having a compiler for. Returns a name -> tokens map and
    the running totals, so the caller can report them rather than trust them.
    """
    out, pic, aud = {}, 0, 0
    for i, c in enumerate(doc["cast"], 1):
        pics = [pic + k + 1 for k in range(int(c.get("pictures", 0) or 0))]
        pic += len(pics)
        auds = [aud + k + 1 for k in range(int(c.get("audio", 0) or 0))]
        aud += len(auds)
        out[c["name"]] = {"subject": i, "pictures": pics, "audio": auds}
    return out, pic, aud


def _sub(text, idx):
    """@name -> <Subject n>, everywhere it appears in free text."""
    def repl(m):
        got = idx.get(m.group(1))
        return f"<Subject {got['subject']}>" if got else m.group(0)
    return NAME.sub(repl, text or "")


def _pictures(nums):
    toks = [f"<Picture {n}>" for n in nums]
    if len(toks) > 1:
        return ", ".join(toks[:-1]) + " and " + toks[-1]
    return toks[0] if toks else ""


def emit(doc):
    """The document -> the fields H3 Long-Form Links and H3 Dialogue take."""
    idx, npic, naud = index(doc)

    defs, rets = [], []
    for c in doc["cast"]:
        got = idx[c["name"]]
        tok = f"<Subject {got['subject']}>"
        if c["kind"] == "character":
            who = c.get("character") or c["name"]
            desc = f"{tok} is {who}"
            if got["pictures"]:
                desc += f", shown in {_pictures(got['pictures'])}"
            if got["audio"]:
                desc += f", whose voice is <Audio {got['audio'][0]}>"
            desc += "."
        elif c["kind"] == "setting":
            desc = f"{tok} is the setting: {_sub(c['description'], idx)}"
        else:
            desc = f"{tok} is {_sub(c['description'], idx)}"
        if not desc.rstrip().endswith((".", "!", "?")):
            desc += "."
        defs.append(desc)

        # RETENTION IS ABOUT LIKENESS. It scopes what stays the same about a
        # subject between shots; it is not where pose or motion is directed.
        # Those belong in the shot description, which is where `note` and `do`
        # already put them. An earlier version of this wrote pose instructions
        # into the retention block to stop a take coming out static; that
        # conflated two fields and encoded a guess as a rule. The take that
        # worked used the plain form below, citing its pictures.
        r = c.get("retention", "partially_preserved")
        detail = (c.get("retention_detail") or "").strip()
        if detail:
            note = _sub(detail, idx)
        elif r == "fully_preserved":
            # no possessive prepended: the author's own text supplies one
            # ("her face"), and adding "their" in front produced "preserve
            # their her face"
            what = c.get("preserve") or ("facial identity, hair, eye colour "
                                         "and build")
            note = f"preserve {what}"
            if got["pictures"]:
                note += f" as shown in {_pictures(got['pictures'])}"
            allow = (c.get("allow") or "").strip()
            note += f", while allowing {allow}." if allow else "."
        elif r == "free":
            note = "no constraint."
        else:
            what = c.get("preserve") or "their appearance"
            note = f"keep {what} consistent from shot to shot."
        rets.append(f"{tok}: {r} - {note}")

    body = []
    for n, s in enumerate(doc["shots"], 1):
        part = f"[Shot {n}] {_sub(s['description'], idx)}"
        if not part.rstrip().endswith((".", "!", "?")):
            part += "."
        for note in s["notes"]:
            t = _sub(note, idx)
            part += " " + (t if t.rstrip().endswith((".", "!", "?")) else t + ".")
        body.append(part)

    # H3Dialogue's own input format: rows of `speaker | verb | line`, blank
    # lines separating chunks; actions one per chunk in the same order.
    lines, actions, speakers = [], [], {}
    for s in doc["shots"]:
        for ch in s["chunks"]:
            for ln in ch["lines"]:
                sid = speakers.setdefault(
                    ln["who"], f"S{len(speakers) + 1}")
                lines.append(f"{sid} | {ln['verb']} | {ln['line']}")
            act = " ".join(a.rstrip(".") + "." for a in ch["actions"])
            actions.append(act)
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()

    smap = "; ".join(f"{sid}=<Subject {idx[nm]['subject']}>"
                     for nm, sid in speakers.items())

    loras = []
    for s in doc["shots"]:
        for l in s["loras"]:
            loras.append(f"# {l['name']} {l['strength']} — add a time span")

    return {
        "head": "\n".join(body),
        "subject_defs": "\n".join(defs),
        "retention": "\n".join(rets),
        "soundscape": _sub(doc.get("soundscape", ""), idx),
        "music": doc.get("music", "N/A"),
        "dialogue_lines": "\n".join(lines),
        "dialogue_actions": "\n".join(actions),
        "speaker_map": smap,
        "lora_schedule": "\n".join(loras),
        "counts": {"subjects": len(doc["cast"]), "pictures": npic,
                   "audio": naud},
        "index": idx,
    }


def lint(doc, out):
    """Things that render fine and are wrong. Report, never refuse."""
    notes = []
    named = set()
    for s in doc["shots"]:
        named |= set(NAME.findall(s["description"]))
        for n in s["notes"]:
            named |= set(NAME.findall(n))
        for ch in s["chunks"]:
            named |= {l["who"] for l in ch["lines"]}
    for c in doc["cast"]:
        if c["name"] not in named:
            notes.append(f"@{c['name']} is declared but never used — it still "
                         f"takes a <Subject> slot")
        if c["kind"] == "character" and not c.get("pictures"):
            notes.append(f"@{c['name']} is a character with no pictures; set "
                         f"@{c['name']}.pictures so it gets <Picture> slots")
    if not doc["shots"]:
        notes.append("no shots — the prompt will have no description")
    for s in doc["shots"]:
        empty = [i + 1 for i, ch in enumerate(s["chunks"])
                 if not ch["lines"] and not ch["actions"]]
        if empty and len(s["chunks"]) > 1:
            notes.append(f"chunk break(s) {empty} have neither a line nor an "
                         f"action; that stretch has nothing directing it")
    return notes


class H3Script:
    """Write a take once, in names, and let the compiler do the numbering."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script": ("STRING", {"multiline": True, "default":
                       "@ada  = character Ada\n"
                       "@man  = a man in his thirties, dark hair\n"
                       "@room = setting. a bedroom in warm low light\n"
                       "\n"
                       "soundscape = room tone, breathing, the bed moving\n"
                       "\n"
                       "shot | a locked-off two-shot of @ada and @man on the bed\n"
                       "  note @ada is on her back, nearer camera\n"
                       "  say  @ada moans Oh god, yes.\n"
                       "  do   his hips work in a steady rhythm\n"}),
        }, "optional": {
            # APPENDED. These two make the emitted `plan` the SAME plan the
            # timeline drew — the board cannot describe one render while ComfyUI
            # performs another, because both come from this call.
            "chunk_frames": ("INT", {"default": 141, "min": 5, "max": 3600,
                             "step": 17,
                             "tooltip": "How long a chunk runs, for shots that "
                                        "declare no length of their own. Sizes "
                                        "the plan this node emits."}),
            "context": ("INT", {"default": 39, "min": 0, "max": 4096,
                        "tooltip": "The handle carried from the previous chunk. "
                                   "Those frames are reproduced under a denoise "
                                   "mask of 0, so nothing can be SAID in them — "
                                   "39 is the smallest count landing on both the "
                                   "24 fps and 40 Hz grids."}),
        }}

    # APPENDED, per the slot contract: cut_frames and total_frames go last so
    # every saved graph keeps its wiring. They are what makes a shot's length
    # mean something downstream -- wire them into H3 Chunk Plan and the cuts you
    # drew are the cuts it plans.
    RETURN_TYPES = ("STRING",) * 8 + ("STRING", "STRING", "STRING", "INT",
                                     "H3_CHUNK_PLAN", "STRING")
    RETURN_NAMES = ("head", "subject_defs", "retention", "soundscape", "music",
                    "dialogue_lines", "dialogue_actions", "speaker_map",
                    "document", "info", "cut_frames", "total_frames", "plan",
                    "lora_schedule")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Compile a take written in @names into H3's prompt fields. "
                   "Feeds H3 Long-Form Links and H3 Dialogue; the numbering is "
                   "worked out for you.")

    def go(self, script, chunk_frames=141, context=39):
        doc = parse(script)
        out = emit(doc)
        notes = lint(doc, out)

        idx = out["index"]
        rows = ["H3 SCRIPT — {subjects} subject(s), {pictures} picture(s), "
                "{audio} audio".format(**out["counts"])]
        for c in doc["cast"]:
            g = idx[c["name"]]
            pics = ",".join(str(p) for p in g["pictures"]) or "-"
            auds = ",".join(str(a) for a in g["audio"]) or "-"
            rows.append(f"  @{c['name']:<12} Subject {g['subject']}   "
                        f"Picture {pics:<8} Audio {auds:<5} {c['kind']}")
        if notes:
            rows.append("  lint:")
            rows += [f"    {n}" for n in notes]
        cuts, total = cuts_from(doc, chunk_frames)
        chunks, plan_info = build_plan(doc, chunk_frames=chunk_frames,
                                       context=context)
        rows += [f"  {n}" for n in audio_clock_notes(doc, chunk_frames)]
        sched, slots, lora_notes = lora_schedule(doc, chunks)
        if slots:
            # the slot map is the part you cannot guess: H3 Chunk Lora addresses
            # a PICKER SLOT, so the schedule is meaningless until lora_1..3 are
            # set to these files
            rows.append("  lora slots — set these on H3 Chunk Lora:")
            rows += [f"    {k} = {v}" for k, v in slots.items()]
        rows += [f"  {n}" for n in lora_notes]
        if cuts:
            rows.append(f"  cuts at {', '.join(str(c) for c in cuts)} "
                        f"of {total} frames — wire cut_frames into H3 Chunk Plan")
        info = "\n".join(rows)

        return {"ui": {"h3char": [info]},
                "result": (out["head"], out["subject_defs"], out["retention"],
                           out["soundscape"], out["music"],
                           out["dialogue_lines"], out["dialogue_actions"],
                           out["speaker_map"],
                           json.dumps(doc, indent=2), info,
                           ",".join(str(c) for c in cuts), total, chunks,
                           sched)}


NODE_CLASS_MAPPINGS = {"H3Script": H3Script}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Script": "H3 Script (write a take in @names)"}
