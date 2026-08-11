"""H3 Prompt Lint — check a prompt against the traps that actually cost us time.

Every rule here is something we hit, diagnosed and paid for. The findings live in
~/projects/h3/docs/prompting-ref2va.md and docs/long-form-waves.md; this makes them
executable instead of tribal, and it works on any prompt string — hand-written,
built by the H3 Audio Prompt node, or pasted from somewhere else.

Severity means what it says:
    ERROR  the model will do something you did not ask for. Fix before rendering.
    WARN   likely to bite, or depends on intent.
    INFO   style / worth knowing.

Deliberately NOT checked: anything needing the render itself. Whether identity
holds, whether a join reads as a cut, whether a voice matches — those are eye
calls, and our own metrics were wrong about them repeatedly.
"""

import re

TASK_TYPES = ["keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"]
SECTIONS = ["subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music"]
VISUAL_MARKERS = ["fully_preserved", "partially_preserved", "attribute_transfer",
                  "weak_reference"]
AUDIO_MARKERS = ["fully_copy", "partially_copy", "reference", "weak_reference"]

# phrasing that assumes the model remembers a previous generation. It does not:
# every prompt is independent, and relative phrasing is what made the early
# long-form chains incoherent (waves 1-5) before self-contained prompts fixed it.
RELATIVE = [
    r"\bcontinuing (?:the|from)\b", r"\bcontinues? from\b", r"\bas (?:before|above)\b",
    r"\bthe same (?:shot|scene|room|setting|clip)\b", r"\bsame as (?:the )?previous\b",
    r"\bpicking up (?:where|from)\b", r"\bresum(?:es|ing) (?:the|from)\b",
    r"\bpreviously\b", r"\bearlier (?:shot|clip|scene)\b", r"\bstill in the\b",
]
# prose timing instead of the trained [Shot N] At MM:SS.mmm form
PROSE_TIME = [
    r"\bfrom \d+[:.]\d+ to \d+[:.]\d+", r"\baround \w+ seconds? in\b",
    r"\bat the \w+[- ]second mark\b", r"\bafter (?:about )?\w+ seconds?\b",
    r"\bhalfway through\b", r"\bnear the (?:start|end|beginning)\b",
]
INSTRUMENTS = ["guitar", "ukulele", "piano", "bass", "drums", "violin", "cello",
               "saxophone", "trumpet", "synth", "keyboard", "banjo", "harmonica"]
# Naming the failure produces it. `090` forbade the anchor appearing as a shot
# and forbade cuts; it cut twice. `091` deleted both sentences, changed nothing
# else, and was clean at two seeds — one of them the seed that had cut. The
# working 14-clip one-take never uses the word: scene changes are staged
# physically, through doors and corridors.
NAMES_FAILURE = [
    r"\bmust not appear\b", r"\bnot frames of the target\b", r"\bno cuts?\b",
    r"\bno camera movement\b", r"\bno angle change\b", r"\bdo(?:es)? not cut\b",
    r"\bwithout (?:any )?cuts?\b", r"\bavoid cut", r"\bnever cuts?\b",
    r"\bno jump cuts?\b", r"\bmust not (?:be shown|be rendered|appear as)\b",
]


def _section(text, name):
    """Body of one labelled section, or None."""
    m = re.search(rf"(?im)^\s*{name}\s*:\s*(.*?)(?=^\s*(?:{'|'.join(SECTIONS)})\s*:|\Z)",
                  text, re.S)
    return m.group(1).strip() if m else None


def lint(prompt, long_form=False):
    """-> [(severity, rule, message)]"""
    out = []
    def err(rule, msg): out.append(("ERROR", rule, msg))
    def warn(rule, msg): out.append(("WARN", rule, msg))
    def info(rule, msg): out.append(("INFO", rule, msg))

    text = prompt or ""
    if not text.strip():
        return [("ERROR", "empty", "Prompt is empty.")]

    low = text.lower()
    six = {s: _section(text, s) for s in SECTIONS}
    has_sections = sum(v is not None for v in six.values()) >= 3
    refs_used = sorted(set(re.findall(r"<(Picture|Video|Audio)\s+(\d+)>", text)))
    body = six["detailed_description"] or six.get("integrated") or text

    # --- references must be declared, typed and given a retention marker
    if refs_used:
        if not has_sections:
            err("format/sections",
                f"Uses {len(refs_used)} reference tag(s) but is not in the six-section "
                f"format. With a reference present the model needs subject_definitions, "
                f"a task-type prefix in summary, and retention_analysis — without them "
                f"the reference has no declared role and the model improvises.")
        else:
            if six["subject_definitions"] is None:
                err("refs/undefined", "References used but there is no subject_definitions "
                                      "section binding them.")
            else:
                for kind, n in refs_used:
                    if f"<{kind} {n}>" not in six["subject_definitions"]:
                        warn("refs/unbound",
                             f"<{kind} {n}> is used but never bound in subject_definitions.")
            if six["retention_analysis"] is None:
                err("refs/no-retention",
                    "References used but no retention_analysis. Each one needs a marker: "
                    f"visual {'/'.join(VISUAL_MARKERS)}; audio {'/'.join(AUDIO_MARKERS)}.")
            else:
                ra = six["retention_analysis"].lower()
                for kind, n in refs_used:
                    if f"<{kind.lower()} {n}>" not in ra:
                        warn("refs/no-marker",
                             f"<{kind} {n}> has no entry in retention_analysis.")
                if any(k == "Audio" for k, _ in refs_used) and "fully_copy" in ra:
                    warn("audio/fully_copy",
                         "fully_copy REUSES the audio signal. For a voice saying NEW "
                         "words you want `reference` — timbre and delivery only.")
                # retention must be claimed BY the subject, not BY the picture
                bound_to_pic = [ln for ln in six["retention_analysis"].splitlines()
                                if re.match(r"\s*<Picture\s+\d+>", ln)]
                if bound_to_pic:
                    err("retention/picture-bound",
                        f"{len(bound_to_pic)} retention_analysis entr(y/ies) name "
                        f"<Picture N> as the retained thing. Measured: with motion context "
                        f"present this makes the model render the anchor image AS A SHOT — "
                        f"a hard cut to the anchor's own background mid-clip (090: cuts at "
                        f"7.71s/9.79s, gone at 091 with the same seed). Bind retention to "
                        f"the SUBJECT and cite the pictures only as the source of her "
                        f"attributes: '<Subject 1> (appears in [Shot 1]): fully_preserved - "
                        f"preserve her facial identity, hairstyle, body proportions ... from "
                        f"<Picture 1> and <Picture 2> while allowing natural poses.'")
        s = six["summary"] or ""
        if not any(f"[{t}]" in s for t in TASK_TYPES):
            err("summary/no-task-type",
                "summary has no task-type prefix. One of: "
                + ", ".join(f"[{t}]" for t in TASK_TYPES))

    # --- the documented task-type trap
    if "[video editing]" in low and re.search(r"\b(continu|extend|resum|carries on)", low):
        err("task/editing-vs-continuation",
            "[video editing] with continuation language. Measured: [video editing] makes "
            "the model RE-RENDER the same action instead of continuing it. Use "
            "[video continuation] to extend a clip.")

    # --- dialogue mechanics
    for m in re.finditer(r"<d>\s*(?!\[)", text):
        err("dialogue/no-language",
            "<d> block without a language tag. The form is <d>[English] text</d>.")
        break
    if re.search(r"</d>\s*\(S\d+\)", text):
        err("dialogue/speaker-after-tag",
            "(Sx) placed AFTER </d>. The trained form is subject + (Sx) + speech verb + "
            "tag; trailing it risks the id being vocalised as a spoken artifact.")
    quoted = re.findall(r'(?<!\[)"([A-Z][^"]{6,})"', text)
    if quoted and "<d>" not in text:
        warn("dialogue/untagged",
             f"Quoted speech but no <d> tags — {len(quoted)} line(s) will NOT be "
             f"vocalised. Only text inside <d>[Lang] ... </d> is spoken.")

    # --- naming the failure produces it
    named = sorted({re.search(p, low).group(0) for p in NAMES_FAILURE
                    if re.search(p, low)})
    if named:
        err("longform/names-failure",
            f"Prompt names the failure it is trying to prevent: {', '.join(named)}. "
            f"Measured: `090` said \"must not appear as shots in it\" and \"no cuts, no "
            f"camera movement, no angle change\" and cut twice mid-clip; `091` deleted "
            f"those sentences with everything else identical and was clean at two seeds, "
            f"one of them the seed that had cut. Say what the shot DOES instead — the "
            f"camera holds, the take runs unbroken — and stage scenery changes physically "
            f"(a door, a corridor) rather than forbidding them.")

    # --- diegetic vs non-diegetic
    ndm = (six["non_diegetic_music"] or "").strip()
    if ndm and ndm.lower() not in ("n/a", "na", "none", ""):
        if any(i in ndm.lower() for i in INSTRUMENTS) and \
           re.search(r"\b(plays|playing|strums|fingerpick|sings while)\b", low):
            err("audio/diegetic-confusion",
                "non_diegetic_music names an instrument while a performer is playing one. "
                "non_diegetic_music is score the characters CANNOT hear — an instrument "
                "someone plays is diegetic and belongs in the description. As written the "
                "model lays a SECOND, unrelated track under the performance.")
        if "<d>" in ndm:
            err("audio/dialogue-in-music",
                "Dialogue tags inside non_diegetic_music. Never repeat dialogue there.")

    # --- timing
    shots = re.findall(r"\[Shot\s+(\d+)\]\s*At\s+(\d+):(\d+(?:\.\d+)?)", text)
    if any(re.search(p, low) for p in PROSE_TIME):
        warn("timing/prose",
             "Prose time window found. [Shot N] At MM:SS.mmm is the trained form; prose "
             "windows are paced far more loosely.")
    if shots:
        times = [int(a) * 60 + float(b) for _, a, b in shots]
        gaps = [(times[i + 1] - times[i], times[i]) for i in range(len(times) - 1)]
        big = [(g, t) for g, t in gaps if g > 6.0]
        for g, t in big[:3]:
            warn("timing/unclaimed",
                 f"{g:.1f}s of unclaimed time after the shot at {t:.1f}s. Unclaimed time "
                 f"gets filled with invented speech/sound — give the gap its own shot "
                 f"marker saying nothing happens.")
        if sorted(times) != times:
            err("timing/out-of-order",
                "[Shot N] timestamps are not in ascending order.")

    # --- long-form specific
    if long_form:
        hits = [p for p in RELATIVE if re.search(p, low)]
        if hits:
            err("longform/relative",
                f"Relative phrasing found ({len(hits)} pattern(s)). Each link is generated "
                f"INDEPENDENTLY — the model carries no memory of the previous prompt. "
                f"Relative wording is what made the early chains incoherent; every link "
                f"must describe the whole scene from scratch.")
        if not re.search(r"\b(single|one) (?:uninterrupted |unbroken )?continuous take\b|"
                         r"\bruns unbroken\b|\bholds? the same framing\b", low):
            warn("longform/no-continuity-clause",
                 "No positive continuity clause. State what the camera DOES — 'a single "
                 "continuous take', 'the camera holds the same framing', 'the take runs "
                 "unbroken' — rather than what it must not do (see longform/names-failure).")

    # --- length guidance (video generation only; the 350-500 figure is for shot
    # description, and an audio-only take has no composition to describe)
    visual = any(k in ("Picture", "Video") for k, _ in refs_used) or \
        re.search(r"\b(camera|dolly|pan|tilt|close[- ]up|wide shot|framing)\b", low)
    words = len(re.findall(r"\w+", body))
    if not visual:
        pass
    elif has_sections and words < 120:
        info("length/short",
             f"detailed_description is {words} words. The guide asks 350-500 for "
             f"generation tasks; short descriptions leave more to chance.")
    elif words > 700:
        info("length/long", f"detailed_description is {words} words — past the 350-500 "
                            f"the guide suggests.")
    return out


class H3PromptLint:
    """Check an H3 prompt against the documented traps before you spend a render."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
                "long_form": ("BOOLEAN", {"default": False,
                              "tooltip": "Also apply the chained-clip rules: no relative "
                                         "phrasing, and a continuity clause present."}),
                "fail_on_error": ("BOOLEAN", {"default": False,
                                  "tooltip": "Raise instead of warning, so a bad prompt "
                                             "stops the queue rather than wasting a render."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("prompt", "report", "clean", "issues")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Lint an H3 prompt: reference bindings, task-type prefix, retention "
                   "markers, <d> dialogue form, diegetic/non-diegetic confusion, [Shot N] "
                   "timing and unclaimed gaps, and long-form relative phrasing. Passes the "
                   "prompt through unchanged so it can sit inline before the H3 node.")

    def run(self, prompt, long_form, fail_on_error):
        findings = lint(prompt, long_form)
        errors = [f for f in findings if f[0] == "ERROR"]
        if not findings:
            report = "clean - no issues found"
        else:
            order = {"ERROR": 0, "WARN": 1, "INFO": 2}
            findings.sort(key=lambda f: order[f[0]])
            report = "\n".join(f"{sev:5s} {rule}\n      {msg}" for sev, rule, msg in findings)
            counts = {s: sum(1 for f in findings if f[0] == s) for s in ("ERROR", "WARN", "INFO")}
            report = (f"{counts['ERROR']} error(s), {counts['WARN']} warning(s), "
                      f"{counts['INFO']} note(s)\n\n" + report)
        if fail_on_error and errors:
            raise ValueError("H3 prompt lint failed:\n" + "\n".join(
                f"{r}: {m}" for _, r, m in errors))
        return {"ui": {"h3lint": [report]},
                "result": (prompt, report, len(errors) == 0, len(findings))}


NODE_CLASS_MAPPINGS = {"H3PromptLint": H3PromptLint}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PromptLint": "H3 Prompt Lint"}
