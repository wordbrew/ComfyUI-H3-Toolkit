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
    @name.wears     = a charcoal wool coat, collar turned up
    @name.voice     = a low measured voice
    @name.pronoun   = she        how later lines in a shot refer back to them

  WARDROBE FOLLOWS THE PROMPT, NOT THE ANCHORS (tested 2026-08-05), and a prompt
  that says nothing about clothing lets the model fill the gap from the reference
  images -- which is what drifted earlier long-form takes topless. `wears` is
  therefore worth stating on every clothed subject. `voice` is the timbre phrase
  the guide puts in front of the speaker id.

  RETENTION IS ABOUT LIKENESS, and only that. `preserve` and `allow` scope what
  stays the same about a subject between shots. Pose, motion and performance go
  in the shot -- `note` and `do` -- not here.

    task       = reference generation | video editing
    style      = the look, once: format, lens, grain, lighting, set, camera
                 behaviour, and what must NOT appear. It opens
                 detailed_description, which the guide wants at 350-500 words
                 for a generation task -- without it a take emits about 130.
    camera     = how the camera behaves across the take: movement type,
                 amplitude and speed, in natural language ("handheld with small
                 continuous drift from the operator's breathing, never a
                 deliberate move"). The guide asks for all three.
    negatives  = what must NOT appear ("no other people, no readable text, no
                 signage, no costume changes, no music"). Stated positively as
                 exclusions, which is the one place a negative belongs.
    lips       = lip discipline, for any take with more than one speaker. Omit
                 it and a listening face mouths the other person's line. Set to
                 `auto` for the default wording, or write your own.
    soundscape = the acoustic SPACE as a sentence, not a list of noises: the
                 worked example describes the room's tail and hum, not "rain,
                 hum, footsteps".
    music      = ...

    shot | <description>
      note <text>                       adds to the description
      say  @name [@seconds] [verb] <line>
                                        dialogue. `@2.5` pins it to 2.5s on the
                                        FINISHED clip -- what the ruler shows.
                                        Without one it flows after the line
                                        before, as H3 Dialogue has always done.
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
    """(anchor count, voice count, description, retention, kind) from the store.

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
            return 0, 0, "", "", "person"
        img = os.path.join(d, "images")
        n = len([f for f in sorted(os.listdir(img))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                ) if os.path.isdir(img) else 0
        voice = 1 if os.path.exists(os.path.join(d, "voice.wav")) else 0
        desc, ret, kind = "", "", "person"
        card = os.path.join(d, "card.json")
        if os.path.exists(card):
            with open(card, encoding="utf-8") as fh:
                data = json.load(fh)
            desc = (data.get("description") or "").strip()
            ret = (data.get("retention") or "").strip()
            # cards written before kinds existed are people, which is what they
            # were
            kind = (data.get("kind") or "person").strip()
        return n, voice, desc, ret, kind
    except Exception:
        return 0, 0, "", "", "person"


# WHAT "KEEP IT THE SAME" MEANS DEPENDS ON WHAT IT IS. A saved lamp used to come
# back as "preserve facial identity, hair, eye colour and build", because the
# store held people and nothing else. These are the defaults; `@name.preserve`
# still overrides any of them.
PRESERVE_BY_KIND = {
    "person":  "facial identity, hair, eye colour and build",
    "place":   "layout, architecture, materials and quality of light",
    "setting": "layout, architecture, materials and quality of light",
    "thing":   "shape, proportions, colour, material and markings",
}

MAX_PICTURES = 4       # the reference node's ref_image slots
MAX_VOICES = 2


def cast_media(doc):
    """The pictures and voices the script's own numbering refers to.

    THE GAP THIS CLOSES. The script decides that @skye is <Subject 1> and holds
    <Picture 1>..<Picture 3>, and then the pictures actually encoded came from
    LoadImage nodes wired separately to the reference node. Nothing checked they
    were the same images -- so the prompt could cite three anchors of one person
    while three pictures of somebody else went to the model. That is the "five
    nodes that have to agree" failure, still standing in the one place the
    compiler had not reached.

    Pulled from the store in the SAME order `index()` assigns, so the picture
    cited and the picture shown cannot come apart.

    -> (images, voices, notes). Torch-free until it actually loads something, so
    the module still imports without ComfyUI.
    """
    images, voices, notes = [], [], []
    for entry in doc.get("cast", []):
        if entry.get("kind") != "character":
            continue
        name = entry.get("character") or entry["name"]
        try:
            import os

            from .character import IMAGE_EXT, _load_audio, _load_image, characters_dir
            d = os.path.join(characters_dir(), name)
            img_dir = os.path.join(d, "images")
            found = []
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.lower().endswith(IMAGE_EXT):
                        found.append(os.path.join(img_dir, f))
            want = int(entry.get("pictures") or len(found))
            for path in found[:want]:
                images.append(_load_image(path))
            if int(entry.get("audio") or 0):
                for cand in ("voice.wav", "voice.flac", "voice.mp3"):
                    vp = os.path.join(d, cand)
                    if os.path.isfile(vp):
                        voices.append(_load_audio(vp))
                        break
        except Exception as exc:      # no store, no ComfyUI, no PIL
            notes.append(f"@{entry['name']}: could not read the store ({exc})")
    if len(images) > MAX_PICTURES:
        notes.append(f"{len(images)} anchors across the cast, but the reference "
                     f"node takes {MAX_PICTURES} — the rest are cited in the "
                     f"prompt and not shown, which is worse than not citing them")
    return images[:MAX_PICTURES], voices[:MAX_VOICES], notes


class ScriptError(ValueError):
    """A problem with the script, reported with the line it is on."""


def _err(lineno, text, line):
    return ScriptError(f"line {lineno}: {text}\n    {line.strip()}")


def parse(text):
    """The DSL -> the document. Pure data; no ComfyUI, no torch."""
    doc = {"version": 1, "cast": [], "shots": [],
           "task": "reference generation", "soundscape": "", "music": "N/A",
           # THE LOOK, once, at the top of detailed_description. The guide asks
           # 350-500 words for a generation task and we were emitting 134 --
           # because the format, the camera, the lighting, the set and the
           # exclusions had nowhere to live. Every worked example opens with
           # them.
           "style": "", "camera": "", "negatives": "", "lips": ""}
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
                    elif attr in ("preserve", "allow", "retention_detail",
                                  "wears", "voice", "pronoun"):
                        # `preserve` lists what stays the same, `allow` what may
                        # vary -- both LIKENESS scoping. `retention_detail`
                        # replaces the whole clause when the default wording is
                        # not what a shot needs.
                        #
                        # `wears` and `voice` are the two things the worked
                        # example states about every speaking subject and the
                        # script had nowhere to put. Wardrobe follows the PROMPT,
                        # not the anchors (tested 2026-08-05) -- and a prompt
                        # that says nothing about clothing lets the model fill
                        # the gap from the reference images, which is what drifted
                        # earlier long-form tests topless. `voice` is the timbre
                        # phrase that rides in front of the speaker id:
                        # "<Subject 1>, with a low measured voice (S1), says:".
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
                    npic, naud, desc, ret, akind = store_card(entry["character"])
                    entry["asset_kind"] = akind
                    entry["pictures"] = npic
                    entry["audio"] = naud
                    if desc:
                        entry["store_description"] = desc
                    if ret in RETENTIONS:
                        # KEEP WHAT THE CARD SAID. Without it an override is
                        # undetectable after parsing -- the script's value has
                        # already replaced it -- so nothing could report that
                        # two sources disagreed, which is the failure this
                        # compiler exists to end. Derived, never serialized.
                        entry["store_retention"] = ret
                        entry["retention"] = ret
                    entry["from_store"] = bool(npic or desc)
                elif rest.startswith("setting."):
                    entry["kind"] = "setting"
                    entry["asset_kind"] = "place"
                    entry["description"] = rest[len("setting."):].strip()
                doc["cast"].append(entry)
                by_name[nm] = entry
                continue

            key, sep, rest = body.partition("=")
            key = key.strip()
            # `style` is the look. `camera`, `negatives` and `lips` are the three
            # other things the worked example's opening paragraph carries and a
            # single style blob buried: how the camera behaves, what must NOT
            # appear, and the lip discipline that keeps a listening face still.
            # Separate keys because each one is a different question to answer,
            # and because lint can then say which is missing.
            if sep and key in ("task", "soundscape", "music", "style",
                               "camera", "negatives", "lips"):
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
            # PLACED BY HAND: `@2.5` is seconds on the FINISHED clip -- the same
            # number the timeline's ruler shows and the same one H3 Dialogue
            # reports, so nothing has to convert between coordinate systems. It
            # was briefly shot-relative, which meant the panel, the document and
            # the emitted line each meant a different thing by the same number.
            # A cast name can never begin with a digit (see NAME), so `@2.5`
            # cannot collide with one.
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


_SETTING_DEFAULTS = {"task": "reference generation", "soundscape": "",
                     "music": "N/A", "style": "", "camera": "", "negatives": "",
                     "lips": ""}
# order matters: parse() reads these off `@name.<attr> = ...` lines
_CAST_ATTRS = ("retention", "pictures", "audio", "preserve", "allow",
               "retention_detail", "wears", "voice", "pronoun")


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


def snap_shot(frames):
    """A shot's length on the legal grid — the PLANNER's grid, not a copy of it.

    `chunkplan.legal_run` is the only implementation of 17n+5 that matters,
    because it is the one the planner applies. The panel used to snap a dragged
    shot in JavaScript, which agreed with this today and had nothing to keep it
    agreeing tomorrow -- the same shape as every row-order and span bug this
    session. The server snaps and the board displays the result.
    """
    from .chunkplan import legal_run
    n = max(5, int(frames))
    down = legal_run(n, "down")
    up = legal_run(n, "up")
    return down if (n - down) <= (up - n) else up


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
        at += snap_shot(shot.get("frames") or chunk_frames)
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


def plan_payload(doc, chunk_frames=141, context=39, mode="fixed",
                 total_frames=None):
    """The plan in the shape H3_CHUNK_PLAN actually travels in.

    H3 Chunk Plan emits `{"chunks": [...], "info": {...}, "total_frames": n}` and
    every consumer unpacks it that way -- `slice_chunk` does `plan.get("chunks")`.
    Emitting the bare list looked right, typechecked as H3_CHUNK_PLAN because the
    socket type is just a label, and died in H3 Chunk Open with
    "'list' object has no attribute 'get'".

    A named type is not a checked one. The shape is the contract.
    """
    chunks, info = build_plan(doc, chunk_frames=chunk_frames, context=context,
                              mode=mode, total_frames=total_frames)
    total = sum(int(c["end"]) - int(c["keep_from"]) for c in chunks) if chunks else 0
    _, from_shots = cuts_from(doc, chunk_frames)
    return {"chunks": chunks, "info": info,
            "total_frames": int(total_frames or from_shots or total)}


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
    """Per-shot `lora` lines -> H3 Chunk Lora's schedule, naming FILES.

    WHAT THE COMPILER IS FOR HERE. The schedule is in time on the FINISHED clip,
    which is not time on the source, because the join drops each chunk's carried
    handle. Writing it by hand means holding that conversion in your head while
    editing shots that move.

    A shot's finished start is the kept frames before it. `chunk_windows` derives
    the same number for dialogue; here it is simply the running total of
    `end - keep_from`.

    IT NAMES THE FILE, NOT THE PICKER SLOT, AND THAT REMOVES THE CEILING.
      An earlier cut emitted `lora_1`..`lora_3` and a slot map you then typed into
      the node's file pickers by hand -- and because there are three pickers, a
      fourth LoRA meant a second node and a block of schedule text pasted into
      it. Both were my invention. `H3ChunkLora` has always accepted a filename
      where a slot name goes (chunklora.py: a row whose name is not a slot is
      loaded by name), and the script already carries picker-exact filenames,
      because that is what `lora <file> <strength>` is written with.

      So: one node, any number of LoRAs, nothing to set by hand. The pickers stay
      for schedules written by hand, where a dropdown beats typing a path.

    -> the schedule text, ready for the node's `schedule` input
    """
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
                name = lora.get("name")
                if not name:
                    continue
                rows.append(f"{stamp_at(start)}-{stamp_at(end)} | {name} | "
                            f"{lora.get('strength', '1.0')}")
        at += n or 0
    return "\n".join(rows)


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
            n = snap_shot(shot.get("frames") or chunk_frames)
            bounds.append((at, at + n))
            at += n

    problems = []

    # ONE PACKER. This used to be its own loop here, which packed from each
    # chunk's floor and knew nothing about `pacing` -- and "spread" is the
    # DEFAULT, so the board showed one schedule and H3 Dialogue produced another
    # the moment a chunk held two lines. Both now call story.place_lines.
    from .story import place_lines

    queue, meta = [], []
    for si, shot in enumerate(doc.get("shots", [])):
        for ch in shot.get("chunks", []):
            for line in ch.get("lines", []):
                text = line.get("line", "")
                dur = syllables(text) / max(0.1, float(syllables_per_second))
                # IDENTITY BY INDEX, not by text. Twelve identical lines are a
                # perfectly ordinary take, and keying on the words collapsed
                # them to one.
                item = {"dur": dur, "text": text, "who": line.get("who", ""),
                        "shot": si, "i": len(queue)}
                if "at" in line:
                    item["at_joined"] = float(line["at"])   # already joined
                queue.append(item)
                meta.append(item)

    placed, place_notes = place_lines(queue, chunks, windows, gap=float(gap))
    problems.extend(n for n in place_notes)

    by_index = {}
    for ci, items in enumerate(placed):
        for it in items:
            by_index[it["i"]] = {
                "shot": it["shot"], "who": it["who"], "line": it["text"],
                "seconds": round(it["dur"], 1), "placed": bool(it.get("pinned")),
                "chunk": ci, "local": round(it["at"], 1),
                "start": round(windows[ci]["offset"] + it["at"], 1),
                "problem": it.get("problem"),
            }
    # the packer stops at the first line that will not fit, so everything after
    # it is unplaced too -- each one is reported rather than silently missing
    out_lines = []
    for it in meta:
        out_lines.append(by_index.get(it["i"], {
            "shot": it["shot"], "who": it["who"], "line": it["text"],
            "seconds": round(it["dur"], 1), "placed": "at_joined" in it,
            "chunk": None, "local": None, "start": None,
            "problem": "no room left in the take"}))
    for rec in out_lines:
        if rec["problem"]:
            problems.append(f"{rec['problem']}: \u201c{rec['line']}\u201d")

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
                    "speech_from": round(w["lo"], 1), "speech_to": round(w["hi"], 1),
                    "starts_at": round(w["offset"] + w["lo"], 1)}
                   for i, (c, w) in enumerate(zip(chunks, windows))],
        "shots": [{"index": i, "start": lo, "frames": hi - lo,
                    "requested": int((doc.get("shots") or [{}])[i].get("frames")
                                     or chunk_frames),
                    "seconds": round((hi - lo) / 24.0, 1)}
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


def _join_and(toks):
    """a, b and c — the list form every section of the guide's examples uses."""
    toks = [t for t in toks if t]
    if len(toks) > 1:
        return ", ".join(toks[:-1]) + " and " + toks[-1]
    return toks[0] if toks else ""


def _pictures(nums):
    return _join_and([f"<Picture {n}>" for n in nums])


def _shots_using(doc):
    """{name: [1-based shot numbers]} — which shots each name appears in.

    Feeds the retention scope, `<Subject 1> (appears in [Shot 1] and [Shot 2])`.
    A marker with no scope leaves the model to guess whether it covers the whole
    take, and a scope naming a shot the prompt does not contain is worse than
    none — which is why the per-chunk path recomputes this against its own
    local numbering rather than reusing the take's.
    """
    out = {}
    for i, s in enumerate(doc.get("shots", []), 1):
        text = " ".join([s["description"]] + list(s["notes"]))
        for ch in s["chunks"]:
            text += " " + " ".join(ch["actions"])
            text += " " + " ".join(l["who"] for l in ch["lines"])
        for nm in set(NAME.findall(text)) | {l["who"] for ch in s["chunks"]
                                             for l in ch["lines"]}:
            out.setdefault(nm, []).append(i)
    return out


def _scope(where, total):
    """The `(appears in [Shot 1] and [Shot 2])` clause, or nothing."""
    if not where or len(where) >= total:
        return ""
    return " (appears in " + _join_and([f"[Shot {i}]" for i in where]) + ")"


def _stamp(frames):
    s = frames / 24.0
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


def _sentence(text, idx=None, cap=True):
    """Author's fragment -> a sentence: substituted, capitalised, stopped."""
    t = _sub(text, idx) if idx is not None else (text or "")
    t = t.strip()
    if not t:
        return ""
    if cap:
        # EVERY sentence, not just the first. An author writing
        # "wider. she turns from the window" gets "Wider. She turns ..." -- the
        # compiler already capitalises the opening, and stopping there produced
        # prose that read like a typo in the middle of a shot.
        t = re.sub(r"(^|[.!?]\s+)([a-z])",
                   lambda m: m.group(1) + m.group(2).upper(), t)
    return t if t.rstrip().endswith((".", "!", "?")) else t + "."


def look_paragraph(doc):
    """The opening paragraph of detailed_description: style, camera, lips, negatives.

    WHY IT IS FOUR FIELDS AND ONE PARAGRAPH. The worked example opens with a
    single block covering format, lens, grain, depth of field, light direction
    and quality, set materials, camera behaviour, lip discipline and exclusions
    -- and every take needs all of it, which is why a script with only `style`
    emitted a look nobody asked for. They are separate keys because each is a
    different question and lint can then name the one you skipped; they join into
    one paragraph because that is the shape the model was trained on.

    The lip line is not decoration. With two speakers and nothing said about it,
    a listening face mouths the other person's words.
    """
    parts = []
    for key in ("style", "camera"):
        got = (doc.get(key) or "").strip()
        if got:
            parts.append(_sentence(got))
    lips = (doc.get("lips") or "").strip()
    if lips.lower() == "auto":
        lips = ("each speaker's lips move only on their own line and are closed "
                "and still while anyone else speaks")
    if lips:
        parts.append(_sentence(lips))
    neg = (doc.get("negatives") or "").strip()
    if neg:
        parts.append(_sentence(neg))
    return " ".join(p for p in parts if p)


def shot_paragraph(doc, idx, n, sid_of, spoken=True, time_base=0,
                   language="English", label=None):
    """One shot as the guide writes it -> (text, frames it covers).

    `[Shot 1]` opens untimed; every later cut carries `At MM:SS.mmm`. That is a
    sentence in the guide, not an example, so it binds.

    `time_base` is what makes a CHUNK possible. A chunk renders a window of the
    take, and its clock starts at its own first frame -- so the same shot that is
    "At 00:05.875" in the whole take is "[Shot 1]", untimed, when it opens the
    chunk being rendered. Handing a chunk the take's clock is how a 5.9-second
    render came to be told about a cut at 5.875 seconds it could never reach.

    DIALOGUE GOES INSIDE ITS SHOT, not after the shot list. The guide's verbatim
    form is subject + (Sx) + speech verb + tag, and the worked example names the
    subject on FIRST mention in a shot and uses a pronoun after -- "She says at
    once:" -- which is also how it avoids reading like a cast list.
    """
    s = doc["shots"][n - 1]
    at_f = sum(int(doc["shots"][i].get("frames") or 0) for i in range(n - 1))
    rel = at_f - time_base
    # LABEL, not index. A chunk's prompt is the only thing the model sees, so its
    # shots have to number from 1 within it -- a lone "[Shot 2]" cites a shot the
    # prompt never describes.
    head_ = f"[Shot {label or n}]"
    if rel > 0:
        head_ += f" At {_stamp(rel)},"
    txt = _sub(s["description"], idx)
    # capitalise only when the header does NOT end in a comma: a timed cut reads
    # "At 00:05.875, wider." and a capital there is wrong
    if txt:
        # a timed cut reads "At 00:05.875, wider." so the FIRST letter stays
        # lower after a comma -- but every sentence inside the fragment still
        # capitalises, or "wider. she turns from the window" ships as written
        lead = not head_.rstrip().endswith(",")
        txt = re.sub(r"(^|[.!?]\s+)([a-z])",
                     lambda m: m.group(1) + m.group(2).upper(), txt)
        if not lead:
            txt = txt[:1].lower() + txt[1:]
    part = f"{head_} {txt}"
    if not part.rstrip().endswith((".", "!", "?")):
        part += "."
    for note in s["notes"]:
        tx = _sentence(note, idx)
        if tx:
            part += " " + tx
    if not spoken:
        return part, int(s.get("frames") or 0)

    said = set()
    for ch in s["chunks"]:
        for ln in ch["lines"]:
            who = ln["who"]
            sid = sid_of.get(who, "S1")
            cast = next((c for c in doc["cast"] if c["name"] == who), {})
            if who in said:
                # a pronoun once the subject is established in this shot. Without
                # a stated one the tag repeats, which reads as a cast list rather
                # than a scene.
                subj = (cast.get("pronoun") or "").strip() or \
                    f"<Subject {idx[who]['subject']}>"
                subj = subj[:1].upper() + subj[1:]
                lead = f"{subj} ({sid})" if subj.startswith("<") else subj
            else:
                said.add(who)
                voice = (cast.get("voice") or "").strip()
                tok = f"<Subject {idx[who]['subject']}>"
                lead = f"{tok}, with {voice} ({sid})," if voice else f"{tok} ({sid})"
            verb = (ln["verb"] or "says").strip().rstrip(",:")
            part += f" {lead} {verb}: <d>[{language}] {ln['line']}</d>"
        for act in ch["actions"]:
            tx = _sentence(act, idx)
            if tx:
                part += " " + tx
    return part, int(s.get("frames") or 0)


def emit(doc):
    """The document -> the fields H3 Long-Form Links and H3 Dialogue take."""
    idx, npic, naud = index(doc)

    # SPEAKER IDS FIRST. The guide assigns them in order of first vocal event and
    # every later section reuses them, so they cannot be a by-product of building
    # the dialogue rows further down.
    sid_of = {}
    for s in doc["shots"]:
        for ch in s["chunks"]:
            for ln in ch["lines"]:
                sid_of.setdefault(ln["who"], f"S{len(sid_of) + 1}")

    shots_using = _shots_using(doc)

    defs, rets = [], []
    for c in doc["cast"]:
        got = idx[c["name"]]
        tok = f"<Subject {got['subject']}>"
        if c["kind"] == "character":
            who = c.get("character") or c["name"]
            # a saved asset's own description beats its bare name: "is skye" tells
            # the model nothing, "is a woman with curly copper-red hair" does
            body_text = (c.get("store_description") or "").strip()
            lead = {"place": "the setting: ", "thing": ""}.get(
                c.get("asset_kind"), "")
            desc = f"{tok} is {lead}{body_text or who}"
            tail_note = ""
            if got["pictures"]:
                desc += f", shown in {_pictures(got['pictures'])}"
                # SAY WHAT THE REFERENCE IS NOT FOR. The worked example does this
                # in every definition -- "The daylight park background of
                # <Picture 2> is not present in the target video." Without it the
                # model has a photograph and no instruction that only the subject
                # is wanted, which is one plausible reading of a reference turning
                # up as a shot of its own.
                if c.get("asset_kind") != "place":
                    tail_note = (f" The background and setting of "
                                 f"{_pictures(got['pictures'])} are not present "
                                 f"in the target video.")
            desc += "." + tail_note
        elif c["kind"] == "setting":
            desc = f"{tok} is the setting: {_sub(c['description'], idx)}"
        else:
            desc = f"{tok} is {_sub(c['description'], idx)}"
        if not desc.rstrip().endswith((".", "!", "?")):
            desc += "."
        # WARDROBE AND VOICE APPLY TO ANYONE, not only to a saved character.
        # They lived in the character branch at first and a described person got
        # neither -- which is the half of the cast most likely to need the
        # wardrobe line, since it has no anchors to fall back on.
        #
        # Clothing follows the PROMPT rather than the anchors (tested
        # 2026-08-05): say nothing and the model fills the gap from the
        # reference images, which is how earlier long-form takes drifted topless.
        if c["kind"] != "setting":
            wears = (c.get("wears") or "").strip()
            if wears:
                desc += f" {tok} wears {wears.rstrip('.')} in the target video."
            if got["audio"]:
                # the guide's own template line: the audio is bound to the
                # SPEAKER ID, not merely attached to the subject
                sid = sid_of.get(c["name"], "S1")
                desc += (f" <Audio {got['audio'][0]}> is the voice for {tok} "
                         f"({sid}).")
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
            what = c.get("preserve") or PRESERVE_BY_KIND.get(
                c.get("asset_kind") or ("setting" if c["kind"] == "setting"
                                        else "person"),
                PRESERVE_BY_KIND["person"])
            note = f"preserve {what}"
            if got["pictures"]:
                note += f" as shown in {_pictures(got['pictures'])}"
            allow = (c.get("allow") or "").strip()
            note += f", while allowing {allow}." if allow else "."
        elif r == "free":
            note = "no constraint."
        else:
            # "keep their appearance consistent" said the same vague thing about a
            # person, a room and a lamp. The guide's own examples are concrete and
            # cite the pictures: "retain the same face, white swept-back hair ...".
            what = c.get("preserve") or PRESERVE_BY_KIND.get(
                c.get("asset_kind") or ("setting" if c["kind"] == "setting"
                                        else "person"),
                PRESERVE_BY_KIND["person"])
            note = f"retain the same {what}"
            if got["pictures"]:
                note += f" as shown in {_pictures(got['pictures'])}"
            note += "."
        # WHICH SHOTS. The guide's form is `<Subject 1> (appears in [Shot 1] and
        # [Shot 2]): marker - ...`, and a marker with no scope leaves the model to
        # guess whether it applies to the whole take.
        scope = _scope(shots_using.get(c["name"], []), len(doc["shots"]))
        rets.append(f"{tok}{scope}: {r} - {note}")

    body = [shot_paragraph(doc, idx, n, sid_of, spoken=False)[0]
            for n in range(1, len(doc["shots"]) + 1)]

    # H3Dialogue's own input format: rows of `speaker | verb | line`, blank
    # lines separating chunks; actions one per chunk in the same order.
    lines, actions, speakers = [], [], {}
    for s in doc["shots"]:
        for ch in s["chunks"]:
            for ln in ch["lines"]:
                sid = speakers.setdefault(
                    ln["who"], f"S{len(speakers) + 1}")
                # a hand-placed time rides on the speaker tag: S1@5.9
                tag = sid if ln.get("at") is None else f"{sid}@{float(ln['at']):g}"
                lines.append(f"{tag} | {ln['verb']} | {ln['line']}")
            act = " ".join(a.rstrip(".") + "." for a in ch["actions"])
            actions.append(act)
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()

    # PLACED TIMES HAVE TO SURVIVE INTO THE LINE FORMAT, or dragging a line on
    # the timeline changes the picture and nothing else. `S1@5.9` is joined time
    # on the finished clip -- the same number the board shows -- and H3 Dialogue
    # honours it instead of packing that line.
    smap = "; ".join(f"{sid}=<Subject {idx[nm]['subject']}>"
                     for nm, sid in speakers.items())

    loras = []
    for s in doc["shots"]:
        for l in s["loras"]:
            loras.append(f"# {l['name']} {l['strength']} — add a time span")

    # THE WHOLE-TAKE PROMPT CARRIES ITS DIALOGUE. `head` stays clean because the
    # chunked path builds per-chunk clauses and would double them up; the
    # single-prompt path had no dialogue at all, which is why a take rendered
    # silent when H3 Long-Form Links was out of the graph.
    #
    # ONE BUILDER FOR BOTH. `shot_paragraph` is what the per-chunk path uses too,
    # so the guide's form -- subject + (Sx) + speech verb + tag, `(Sx)` BEFORE
    # the tag, a pronoun after first mention -- exists in one place. The two used
    # to be written separately and had drifted: this path lost the style
    # paragraph, that one lost the shot slicing.
    head = "\n".join(body)
    described = "\n\n".join(
        shot_paragraph(doc, idx, n, sid_of)[0]
        for n in range(1, len(doc["shots"]) + 1))
    look = look_paragraph(doc)
    if look:
        described = look + "\n\n" + described
    defs_s = "\n".join(defs) if defs else "N/A"
    rets_s = "\n".join(rets) if rets else "N/A"
    # THE ASSEMBLED PROMPT, in the block order prompt_scene.py uses. Every
    # consumer -- the reference nodes, H3 Long-Form Links -- wants one string;
    # the separate fields exist so a UI can show them apart, not because anything
    # downstream takes them apart. Without this a workflow has to concatenate
    # them by hand, in the right order, which is the bookkeeping this node exists
    # to remove.
    # THE TASK-TYPE PREFIX IS A RULE, not decoration. The guide names six, and
    # the wrong one changes behaviour -- `[video editing]` on an extension makes
    # the model re-render the same action instead of continuing it. Our own
    # assembled prompt was emitting the head's first line with no prefix at all.
    n_sh = len(doc.get("shots", []))
    who = ", ".join(f"<Subject {idx[c['name']]['subject']}>"
                    for c in doc["cast"] if c["kind"] != "setting")
    # "2 shots with <Subject 1>, using the listed references" is a manifest, not
    # a summary. The guide wants a paragraph saying what happens; the worked
    # example reads "In three shots on a concrete stairwell landing, <Subject 1>
    # and <Subject 2> trade eight short clipped lines about a plan that has
    # moved." Ours is built from what the author actually wrote.
    setting_txt = next((_sub(c["description"], idx) for c in doc["cast"]
                        if c["kind"] == "setting" and c.get("description")), "")
    n_lines = sum(len(ch["lines"]) for s in doc["shots"] for ch in s["chunks"])
    opening = (_sub(doc["shots"][0]["description"], idx).rstrip(" .")
               if doc["shots"] else "")
    summary = (f"[{doc.get('task', 'reference generation')}] "
               + (f"In {n_sh} shots" if n_sh > 1 else "In a single shot")
               + (f" in {setting_txt}" if setting_txt else "")
               + (f", {who}" if who else "")
               + (f" speak{'s' if len(sid_of) == 1 else ''} {n_lines} "
                  f"line{'s' if n_lines != 1 else ''}" if n_lines else "")
               + (f". It opens on {opening}." if opening else ".")
               + (" The references supply identity"
                  + (" and voice." if naud else ".") if npic or naud else ""))
    prompt = (
        "subject_definitions:\n" + defs_s + "\n\n"
        "summary:\n" + summary + "\n\n"
        "retention_analysis:\n" + rets_s + "\n\n"
        "detailed_description:\n" + described + "\n\n"
        "overall_soundscape: " + (_sub(doc.get("soundscape", ""), idx).strip() or "N/A")
        + "\n\n"
        "non_diegetic_music: " + (doc.get("music") or "N/A"))

    # THE TAIL HAS TO KNOW HOW MANY SHOTS THERE ARE. H3 38 carried the default
    # "a single continuous take ... runs unbroken" over a two-shot script, so the
    # prompt argued with itself about whether it cuts. Prompts NAME the failure
    # they cause -- "no cuts" produces cuts -- so a multi-shot take states the
    # cuts positively rather than denying continuity.
    n_shots = len(doc.get("shots", []))
    if n_shots > 1:
        tail = (f"{n_shots} shots in sequence. The camera cuts cleanly between "
                f"them; each shot holds its own framing from its first frame to "
                f"its last, and the subject stays the same person across the "
                f"cuts.")
    else:
        tail = ("A single continuous take from one camera position; the framing "
                "holds and the take runs unbroken from the first frame to the "
                "last.")

    return {
        "prompt": prompt,
        "tail": tail,
        "head": head,
        "subject_defs": defs_s,
        "retention": rets_s,
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


CLAUSE_SEP = "\n---\n"


def chunk_prompts(doc, chunks):
    """One COMPLETE six-section prompt per chunk, joined for H3 Long-Form Links.

    THE FAILURE THIS EXISTS TO REMOVE
      The chunked path used to ship loose sections -- head, tail, defs, retention
      -- to H3 Long-Form Links, which pasted them around a dialogue clause. The
      result (measured 2026-09-12 on H3 38, chunk 0) was that:

        - the style paragraph never arrived at all, because it rode the whole-take
          `prompt` output that the chunked graph does not use;
        - every chunk was handed the WHOLE take's shot list, so a 5.9-second
          chunk was told about a cut at 00:05.875 it could never reach;
        - the shot clock and the dialogue clock were different clocks in the same
          paragraph, later timestamp first, so time ran backwards;
        - the continuity clause landed inside detailed_description as prose about
          structure rather than in summary.

      A chunk is a window on the take, so the honest thing is to describe THAT
      WINDOW: the shots it contains, on its own clock, with the dialogue in them.

    WHICH SHOTS A CHUNK CONTAINS
      A chunk's kept region is `keep_from..end`, not `start..end` -- the frames
      before `keep_from` are the carried handle, reproduced under a denoise mask
      of 0. A shot is included when it overlaps that kept region at all, because
      a shot that merely starts before the window still fills it.

    WHAT STAYS WHOLE
      subject_definitions and retention_analysis are the take's, not the chunk's:
      identity must not drift between chunks, and re-scoping a subject's shot list
      per chunk would say "appears in [Shot 1]" to one chunk and "[Shot 2]" to
      the next about the same person. The look paragraph is repeated in every
      chunk for the same reason -- it is the one thing that must not change.
    """
    idx, npic, naud = index(doc)
    out = emit(doc)
    shots = doc.get("shots", [])
    if not shots or not chunks:
        return out["prompt"]

    # first vocal event decides the ids, take-wide -- the same order emit() uses
    sid_of, order = {}, []
    for s in shots:
        for ch in s["chunks"]:
            for ln in ch["lines"]:
                if ln["who"] not in sid_of:
                    sid_of[ln["who"]] = f"S{len(sid_of) + 1}"
                    order.append(ln["who"])

    bounds, at = [], 0
    for s in shots:
        n = int(s.get("frames") or 0)
        bounds.append((at, at + n))
        at += n
    look = look_paragraph(doc)
    using = _shots_using(doc)

    prompts = []
    for c in chunks:
        lo, hi = int(c.get("keep_from", c["start"])), int(c["end"])
        mine = [i + 1 for i, (a, b) in enumerate(bounds)
                if b > lo and a < hi] or [len(shots)]
        base = bounds[mine[0] - 1][0]
        # numbered from 1 WITHIN THIS CHUNK, and the retention scope renumbered
        # to match -- a scope citing a shot the prompt does not contain is worse
        # than no scope at all.
        local = {n: k + 1 for k, n in enumerate(mine)}
        paras = [shot_paragraph(doc, idx, n, sid_of, time_base=base,
                                label=local[n])[0] for n in mine]
        described = "\n\n".join(paras)
        if look:
            described = look + "\n\n" + described
        rets = []
        for line in out["retention"].splitlines():
            nm = next((c["name"] for c in doc["cast"]
                       if f"<Subject {idx[c['name']]['subject']}>" ==
                       line.split(" (")[0].split(":")[0].strip()), None)
            line = re.sub(r" \(appears in [^)]*\)", "", line, count=1)
            here = [local[n] for n in using.get(nm, []) if n in local]
            head_, sep, rest = line.partition(":")
            rets.append(head_ + _scope(here, len(mine)) + sep + rest)
        retention = "\n".join(rets) if rets else out["retention"]
        # THE CONTINUITY CLAUSE BELONGS IN SUMMARY, not in the shot description,
        # and it states how many cuts THIS WINDOW has -- a different number from
        # the take's, and the one the model is about to render.
        #
        # The shape follows the worked example: "In three shots on a concrete
        # stairwell landing, <Subject 1> and <Subject 2> trade eight short
        # clipped lines...". Count, place, who, how much they say, then how the
        # camera behaves.
        cuts = len(mine)
        # WHO SPEAKS HERE, not who speaks in the take. Naming a subject who says
        # nothing in this window tells the model to find them a line.
        here_who = []
        for n in mine:
            for ch in shots[n - 1]["chunks"]:
                for ln in ch["lines"]:
                    if ln["who"] not in here_who:
                        here_who.append(ln["who"])
        who = _join_and([f"<Subject {idx[n]['subject']}>" for n in here_who])
        n_lines = sum(len(ch["lines"]) for n in mine
                      for ch in shots[n - 1]["chunks"])
        setting_txt = next((_sub(c2["description"], idx) for c2 in doc["cast"]
                            if c2["kind"] == "setting" and c2.get("description")),
                           "")
        bits = ["In a single continuous shot" if cuts == 1 else f"In {cuts} shots"]
        if setting_txt:
            bits.append(f" in {setting_txt.rstrip('.')}")
        if who and n_lines:
            bits.append(f", {who} speak{'s' if len(here_who) == 1 else ''} "
                        f"{n_lines} line{'s' if n_lines != 1 else ''}")
        elif who:
            bits.append(f", {who} appear{'s' if len(here_who) == 1 else ''}")
        summary = (f"[{doc.get('task', 'reference generation')}] "
                   + "".join(bits).rstrip() + ". "
                   + ("The framing holds and the run is unbroken." if cuts == 1
                      else "The camera cuts cleanly between them; each shot "
                           "holds its own framing and the subjects stay the "
                           "same people across the cuts.")
                   + (" The references supply identity"
                      + (" and voice." if naud else ".") if npic or naud else ""))
        prompts.append(
            "subject_definitions:\n" + out["subject_defs"] + "\n\n"
            "summary:\n" + summary + "\n\n"
            "retention_analysis:\n" + retention + "\n\n"
            "detailed_description:\n" + described + "\n\n"
            "overall_soundscape: " + (out["soundscape"].strip() or "N/A") + "\n\n"
            "non_diegetic_music: " + (out["music"] or "N/A"))
    return CLAUSE_SEP.join(prompts)


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

    # THE LOOK PARAGRAPH IS WHAT THE GUIDE ASKS 350-500 WORDS FOR, and a script
    # that skips it emits a skeleton the model fills from the reference images.
    # Each of these is a separate question, so each gets its own note rather than
    # one "the style is thin".
    if not (doc.get("style") or "").strip():
        notes.append("no `style =` — nothing states format, lens, grain, "
                     "lighting or set, so the look comes from the references")
    if not (doc.get("camera") or "").strip():
        notes.append("no `camera =` — the guide wants movement type, amplitude "
                     "and speed; without it the camera invents a move")
    speakers = {l["who"] for s in doc["shots"] for ch in s["chunks"]
                for l in ch["lines"]}
    if len(speakers) > 1 and not (doc.get("lips") or "").strip():
        # measured behaviour, not a style preference: with nothing said about it
        # a listening face mouths the other person's line
        notes.append(f"{len(speakers)} speakers and no `lips =` — set it to "
                     f"`auto`, or a listening face will mouth the other "
                     f"person's line")
    if not (doc.get("negatives") or "").strip():
        notes.append("no `negatives =` — nothing excludes extra people, "
                     "readable text or music")
    for c in doc["cast"]:
        if c["kind"] in ("character", "subject") and c["name"] in named \
                and not (c.get("wears") or "").strip():
            # wardrobe follows the prompt, not the anchors; saying nothing lets
            # the model fill the gap from the reference images
            notes.append(f"@{c['name']} has no `.wears` — clothing will come "
                         f"from the reference images")
    words = len((out.get("prompt") or "").split())
    if words < 350:
        notes.append(f"the whole-take prompt is {words} words; the guide asks "
                     f"350-500 for a generation task")
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
                       "style = Photorealistic live-action, 16:9, handheld with "
                       "visible grain and shallow depth of field, lit warm and "
                       "low from one lamp at frame left. No other people, no "
                       "readable text, no music.\n"
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
    # `chunk_prompts` goes LAST, on the end, per the slot contract. It is the one
    # to wire for a chunked take: a complete six-section prompt per chunk, into
    # H3 Long-Form Links' `beats`, with head/tail/defs/retention left unwired.
    # The loose sections stay for the graphs that already use them.
    RETURN_TYPES = ("STRING",) * 8 + ("STRING", "STRING", "STRING", "INT",
                                     "H3_CHUNK_PLAN", "STRING", "STRING",
                                     "STRING") + \
                   ("IMAGE",) * MAX_PICTURES + ("AUDIO",) * MAX_VOICES + \
                   ("STRING",)
    RETURN_NAMES = ("head", "subject_defs", "retention", "soundscape", "music",
                    "dialogue_lines", "dialogue_actions", "speaker_map",
                    "document", "info", "cut_frames", "total_frames", "plan",
                    "lora_schedule", "prompt", "tail") + \
                   tuple(f"picture_{i + 1}" for i in range(MAX_PICTURES)) + \
                   tuple(f"voice_{i + 1}" for i in range(MAX_VOICES)) + \
                   ("chunk_prompts",)
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
            # SAY WHICH SOURCE WON. The card is a default and the script is a
            # per-run override, and until now the prompt showed the result with
            # nothing saying the two had disagreed -- a silent disagreement
            # between two sources being exactly what this node exists to end.
            keep = c.get("retention", "partially_preserved")
            card = c.get("store_retention")
            if card and card != keep:
                keep = f"{keep} (overriding the card's {card})"
            rows.append(f"  @{c['name']:<12} Subject {g['subject']}   "
                        f"Picture {pics:<8} Audio {auds:<5} {c['kind']:<10} "
                        f"{keep}")
        if notes:
            rows.append("  lint:")
            rows += [f"    {n}" for n in notes]
        cuts, total = cuts_from(doc, chunk_frames)
        payload = plan_payload(doc, chunk_frames=chunk_frames, context=context)
        chunks = payload["chunks"]
        rows += [f"  {n}" for n in audio_clock_notes(doc, chunk_frames)]

        # the cast's own anchors, in the order this node numbered them
        pics, voices, media_notes = cast_media(doc)
        if pics:
            rows.append(f"  picture_1..picture_{len(pics)} carry the cast's own "
                        f"anchors — wire them to the reference node so the "
                        f"pictures CITED are the pictures SHOWN")
        rows += [f"  {n}" for n in media_notes]

        sched = lora_schedule(doc, chunks)
        # NOTHING TO SET BY HAND. The schedule names the files, so one H3 Chunk
        # Lora carries the whole take however many LoRAs it uses -- the pickers
        # stay empty and the three-slot chaining this used to print is gone.
        if sched:
            files = []
            for shot in doc.get("shots", []):
                for lora in shot.get("loras", []):
                    if lora.get("name") and lora["name"] not in files:
                        files.append(lora["name"])
            rows.append(f"  H3 Chunk Lora — {len(files)} LoRA(s) on the "
                        f"lora_schedule wire, pickers stay empty:")
            rows += [f"    {n}" for n in files]
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
                           ",".join(str(c) for c in cuts), total, payload,
                           sched, out["prompt"], out["tail"],
                           *(list(pics) + [None] * MAX_PICTURES)[:MAX_PICTURES],
                           *(list(voices) + [None] * MAX_VOICES)[:MAX_VOICES],
                           chunk_prompts(doc, chunks))}


NODE_CLASS_MAPPINGS = {"H3Script": H3Script}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Script": "H3 Script (write a take in @names)"}
