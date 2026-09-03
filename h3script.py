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

  It also does not read the character store for anchor counts. `pictures` is
  declared, because a compiler that silently renumbers `<Picture n>` when a file
  appears in a folder is worse than one that makes you say the number.

GRAMMAR
    # comment
    @name  = <description>              a subject
    @name  = character <StoreName>      a subject whose anchors are in the store
    @name  = setting. <description>     the setting
    @name.retention = fully_preserved | partially_preserved | free
    @name.preserve  = facial identity, hair, eye colour and build
    @name.allow     = natural movement and changing expression
    @name.retention_detail = <replaces the whole clause>
    @name.pictures  = 3
    @name.audio     = 1

  RETENTION IS ABOUT LIKENESS, and only that. `preserve` and `allow` scope what
  stays the same about a subject between shots. Pose, motion and performance go
  in the shot -- `note` and `do` -- not here. Conflating the two makes the field
  stop meaning what H3's format says it means.

    task       = reference generation | video editing
    soundscape = ...
    music      = ...

    shot | <description>
      note <text>                       adds to the description
      say  @name [verb] <line>          dialogue, in order
      do   <text>                       action for the current chunk
      lora <file> <strength|a -> b>
      ---                               chunk break, as in H3 Dialogue
"""

import json
import re

CATEGORY = "MiniMax H3/prompt"

RETENTIONS = ("fully_preserved", "partially_preserved", "free")
NAME = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


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
                _, _, desc = body.partition("|")
                shot = {"description": desc.strip(), "notes": [],
                        "chunks": [{"lines": [], "actions": []}], "loras": []}
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
            speech_verb = "says"
            for v in ("moans", "asks", "whispers", "says", "shouts", "sighs"):
                if after.startswith(v + " "):
                    speech_verb, after = v, after[len(v):].strip()
                    break
            if not after:
                raise _err(lineno, "say has no line to speak", line)
            cur["lines"].append({"who": who, "verb": speech_verb,
                                 "line": after})
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
                       "@ada.pictures = 3\n"
                       "@man  = a man in his thirties, dark hair\n"
                       "@room = setting. a bedroom in warm low light\n"
                       "\n"
                       "soundscape = room tone, breathing, the bed moving\n"
                       "\n"
                       "shot | a locked-off two-shot of @ada and @man on the bed\n"
                       "  note @ada is on her back, nearer camera\n"
                       "  say  @ada moans Oh god, yes.\n"
                       "  do   his hips work in a steady rhythm\n"}),
        }}

    RETURN_TYPES = ("STRING",) * 8 + ("STRING", "STRING")
    RETURN_NAMES = ("head", "subject_defs", "retention", "soundscape", "music",
                    "dialogue_lines", "dialogue_actions", "speaker_map",
                    "document", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Compile a take written in @names into H3's prompt fields. "
                   "Feeds H3 Long-Form Links and H3 Dialogue; the numbering is "
                   "worked out for you.")

    def go(self, script):
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
        info = "\n".join(rows)

        return {"ui": {"h3char": [info]},
                "result": (out["head"], out["subject_defs"], out["retention"],
                           out["soundscape"], out["music"],
                           out["dialogue_lines"], out["dialogue_actions"],
                           out["speaker_map"],
                           json.dumps(doc, indent=2), info)}


NODE_CLASS_MAPPINGS = {"H3Script": H3Script}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Script": "H3 Script (write a take in @names)"}
