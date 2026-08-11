"""H3 Scene Prompt — assemble the six-section format from a cast and a shot list.

The format is documented in ~/projects/h3/docs/prompting-ref2va.md; this writes it
so you do not have to remember which section a thing belongs in, or that the
speaker tag goes BEFORE the dialogue tag rather than after.

    subject_definitions   who and what the references are   <- from H3 Character
    summary               task-type prefix + each ref's role
    retention_analysis    per-reference marker              <- from H3 Character
    detailed_description  shot by shot, in playback order
    overall_soundscape    diegetic ambience only
    non_diegetic_music    audience-only score, or N/A

SHOT SYNTAX — one shot per line, dialogue on `>` lines beneath it:

    A close-up of <Subject 1> at the window, rain outside. Slow push in.
    > S1: I told you I would come back.
    0:06 | Wider. She crosses the room and sits. Static.
    > S2: You said that last time.

A leading `M:SS |` pins the shot's time; unpinned shots are spread evenly. The
first shot is deliberately untimed — the guide opens [Shot 1] without a stamp.

THINGS THIS GETS RIGHT THAT ARE EASY TO GET WRONG
  - `[Shot N] At MM:SS.mmm` rather than prose windows, which pace loosely.
  - Dialogue as subject + (Sx) + speech verb + <d>[Lang] ...</d>. Trailing the
    speaker id after </d> risks it being vocalised as an artifact.
  - non_diegetic_music defaults to N/A. It is score the characters CANNOT hear —
    an instrument someone is playing is diegetic and belongs in the description,
    and putting it here lays a second unrelated track underneath.
  - `[video continuation]`, not `[video editing]`, for extending a clip. Editing
    makes the model re-render the same action instead of continuing it.

The prompt is linted on the way out, so mistakes surface on the node rather than
after a render.
"""

import re

from .lint import lint

TASK_TYPES = ["reference generation", "keyframe completion", "video continuation",
              "video editing", "audio reference", "audio reuse"]
FPS = 24


def align_frames(seconds):
    n = max(5, round(seconds * FPS))
    while n % 17 != 5:
        n += 1
    return n


def stamp(t):
    return f"00:{t:06.3f}" if t < 60 else f"{int(t) // 60:02d}:{t % 60:06.3f}"


def parse_shots(text):
    """-> [{"at": float|None, "text": str, "lines": [(speaker, text), ...]}]"""
    shots = []
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            m = re.match(r">\s*(S\d+)\s*:\s*(.+)$", line)
            if m and shots:
                shots[-1]["lines"].append((m.group(1).upper(), m.group(2).strip()))
            elif shots:                       # untagged dialogue -> S1
                shots[-1]["lines"].append(("S1", line.lstrip("> ").strip()))
            continue
        m = re.match(r"^(?:(\d+):(\d+(?:\.\d+)?)\s*\|\s*)?(.+)$", line)
        at = (int(m.group(1)) * 60 + float(m.group(2))) if m.group(1) else None
        shots.append({"at": at, "text": m.group(3).strip(), "lines": []})
    n = len(shots)
    for i, s in enumerate(shots):
        if s["at"] is None:
            s["at"] = 0.0 if i == 0 else None
    return shots


def place(shots, seconds):
    """Give every unpinned shot a time, keeping pinned ones exactly where they are."""
    n = len(shots)
    if not n:
        return shots
    for i, s in enumerate(shots):
        if s["at"] is None:
            prev = next((shots[j]["at"] for j in range(i - 1, -1, -1)
                         if shots[j]["at"] is not None), 0.0)
            nxt_i = next((j for j in range(i + 1, n) if shots[j]["at"] is not None), None)
            nxt = shots[nxt_i]["at"] if nxt_i is not None else seconds
            span = nxt_i - (i - 1) if nxt_i is not None else n - (i - 1)
            s["at"] = prev + (nxt - prev) / max(1, span)
    return shots


class H3ScenePrompt:
    """Build a six-section H3 prompt from a cast and a shot list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_type": (TASK_TYPES, {"default": "reference generation",
                              "tooltip": "[video continuation] extends a clip. "
                                         "[video editing] MODIFIES a source and will "
                                         "re-render the same action instead of "
                                         "continuing it."}),
                "seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0,
                                      "step": 0.5}),
                "style": ("STRING", {"multiline": True, "default":
                          "Realistic video, shallow depth of field, warm practical light",
                          "tooltip": "Opening sentence of detailed_description — the look, "
                                     "not the plot."}),
                "shots": ("STRING", {"multiline": True, "default":
                          "A close-up of <Subject 1> at the window, rain running down the "
                          "glass. Slow push in.\n"
                          "> S1: I told you I would come back.\n"
                          "0:04 | Wider. She turns from the window and crosses the room. "
                          "The camera holds still.\n",
                          "tooltip": "One shot per line. `M:SS |` pins a time. Lines "
                                     "starting with `>` are dialogue: `> S1: text`."}),
                "soundscape": ("STRING", {"multiline": True, "default":
                               "Rain on glass, a room tone, distant traffic",
                               "tooltip": "Diegetic ambience only — what exists in the "
                                          "room."}),
                "non_diegetic_music": ("STRING", {"multiline": True, "default": "N/A",
                                       "tooltip": "Score the characters CANNOT hear. Leave "
                                                  "N/A if a character is playing the music "
                                                  "— that is diegetic and belongs in a "
                                                  "shot description."}),
                "language": (["English", "Chinese", "Japanese", "Korean", "Spanish",
                              "French", "German", "Portuguese", "Italian", "Russian"],
                             {"default": "English"}),
            },
            "optional": {
                # typed OR wired. Wire H3 Character for a saved cast member, or just
                # describe someone here — plain prose is expanded into the tag syntax.
                "subject_def_1": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Either full wording (contains <Subject N>) "
                                             "or a plain description like 'a woman with "
                                             "curly copper-red hair'. Plain text is "
                                             "expanded into subject_definitions for you. "
                                             "Wire H3 Character to use a saved one."}),
                "pictures_1": ("INT", {"default": 0, "min": 0, "max": 9,
                               "tooltip": "How many <Picture N> anchors this subject has, "
                                          "when you typed a plain description. Ignored if "
                                          "the text already contains <Subject N>."}),
                "retention_1": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "Leave empty and it is written for you, bound "
                                           "to the SUBJECT rather than to <Picture N>."}),
                "subject_def_2": ("STRING", {"multiline": True, "default": ""}),
                "pictures_2": ("INT", {"default": 0, "min": 0, "max": 9}),
                "retention_2": ("STRING", {"multiline": True, "default": ""}),
                "subject_def_3": ("STRING", {"multiline": True, "default": ""}),
                "pictures_3": ("INT", {"default": 0, "min": 0, "max": 9}),
                "retention_3": ("STRING", {"multiline": True, "default": ""}),
                "extra_direction": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "length", "lint", "plan")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Assemble subject_definitions / summary / retention_analysis / "
                   "detailed_description / overall_soundscape / non_diegetic_music from "
                   "a cast (wire H3 Character) and a shot list. Lints itself on the way "
                   "out.")

    def build(self, task_type, seconds, style, shots, soundscape, non_diegetic_music,
              language, subject_def_1="", retention_1="", subject_def_2="",
              retention_2="", subject_def_3="", retention_3="", extra_direction="",
              pictures_1=0, pictures_2=0, pictures_3=0):
        frames = align_frames(seconds)
        dur = frames / FPS
        raw = [(subject_def_1, retention_1, pictures_1),
               (subject_def_2, retention_2, pictures_2),
               (subject_def_3, retention_3, pictures_3)]
        defs, rets, n_sub = [], [], 0
        for text, ret, pics in raw:
            text, ret = text.strip(), ret.strip()
            if not text:
                continue
            n_sub += 1
            if "<Subject" in text:          # already formatted — H3 Character, or by hand
                defs.append(text)
                if ret:
                    rets.append(ret)
                continue
            # plain prose: write the tag syntax so nobody has to remember it
            subj = f"<Subject {n_sub}>"
            offset = sum(p for _, _, p in raw[:n_sub - 1])
            names = [f"<Picture {offset + i + 1}>" for i in range(pics)]
            plist = (" and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1
                     else (names[0] if names else ""))
            line = f"{subj} is {text.rstrip('.')}."
            if plist:
                line += f" {subj}'s appearance is given by {plist}."
            defs.append(line)
            if ret:
                rets.append(ret)
            elif plist:
                # bound to the SUBJECT, never to <Picture N> — naming the picture as the
                # retained thing makes the model render the anchor as a shot
                rets.append(f"{subj} (appears in [Shot 1]): fully_preserved - preserve "
                            f"their facial identity, hairstyle, colouring and body "
                            f"proportions from {plist}, while allowing natural poses, "
                            f"expressions and framing.")

        shot_list = place(parse_shots(shots), dur)
        plan = []
        body = style.strip().rstrip(".") + "." if style.strip() else ""
        for i, sh in enumerate(shot_list):
            head = ("[Shot 1] " if i == 0
                    else f"[Shot {i + 1}] At {stamp(sh['at'])}, ")
            txt = sh["text"].rstrip(".")
            body += ("" if not body else " ") + f"{head}{txt}."
            for spk, line in sh["lines"]:
                # verbatim form: subject + (Sx) + speech verb + tag. The id must
                # precede the tag or it risks being vocalised.
                subj = f"<Subject {spk[1:]}>" if defs else "The speaker"
                body += f" {subj} ({spk}) says, <d>[{language}] {line}</d>"
            tag = f"{len(sh['lines'])} line(s)" if sh["lines"] else "no dialogue"
            plan.append(f"{sh['at']:6.2f}s  shot {i + 1}: {tag}  {txt[:44]}")
            nxt = shot_list[i + 1]["at"] if i + 1 < len(shot_list) else dur
            if nxt - sh["at"] > 8.0:
                plan.append(f"        ! {nxt - sh['at']:.1f}s on this shot — long for one "
                            f"setup; unclaimed time invites invention")
        if extra_direction.strip():
            body += " " + extra_direction.strip()

        who = ", ".join(re.findall(r"(<Subject \d+>)", "\n".join(defs))[:1] or ["the subject"])
        summary = (f"[{task_type}] The target video shows {who} as described below, "
                   f"using the listed references for identity and voice.")

        ndm = non_diegetic_music.strip() or "N/A"
        prompt = (
            "subject_definitions:\n" + ("\n".join(defs) if defs else "N/A") + "\n\n"
            "summary:\n" + summary + "\n\n"
            "retention_analysis:\n" + ("\n".join(rets) if rets else "N/A") + "\n\n"
            "detailed_description:\n" + body + "\n\n"
            "overall_soundscape: " + (soundscape.strip() or "N/A") + "\n\n"
            "non_diegetic_music: " + ndm)

        findings = lint(prompt)
        if findings:
            order = {"ERROR": 0, "WARN": 1, "INFO": 2}
            findings.sort(key=lambda f: order[f[0]])
            report = "\n".join(f"{s:5s} {r}: {m}" for s, r, m in findings)
        else:
            report = "clean - no issues found"
        plan_text = (f"{len(shot_list)} shot(s)  {dur:.2f}s  {frames} frames\n"
                     + "\n".join(plan))
        return {"ui": {"h3lint": [report], "h3plan": [plan_text]},
                "result": (prompt, frames, report, plan_text)}


NODE_CLASS_MAPPINGS = {"H3ScenePrompt": H3ScenePrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ScenePrompt": "H3 Scene Prompt"}
