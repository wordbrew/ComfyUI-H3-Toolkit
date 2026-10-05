"""H3 Prompt Rewriter — turn a one-line idea into a correct six-section prompt.

Deliberately NOT tied to one LLM integration. It emits a system prompt and a user
prompt as plain STRINGs, so you can feed whatever text generator you already have
(any local LLM node, an API node, or your own hand-editing), then hand the reply to
H3 Rewriter Parse, which normalises the sections and lints the result.

Why a brief rather than a built-in model: the value here is the INSTRUCTIONS — the
format, the traps, the vocabulary — and those are the same regardless of which
model writes the text. Binding this to one LLM node pack would make it fragile and
would not make the output better.

The system prompt encodes what the guide says and what we measured, including the
things a general-purpose model reliably gets wrong: that only <d> text is spoken,
that the speaker id goes before the tag, that an instrument someone plays is
diegetic, and that naming a failure ("no cuts") tends to produce it.
"""

import re

from .prompt_lint import lint

SECTIONS = ["subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music"]

SYSTEM = """You write prompts for MiniMax-H3, a video+audio generation model. \
Output ONLY the prompt, in exactly six labelled sections, in this order and with \
these exact labels:

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

RULES, all of which matter:

1. subject_definitions labels every referenced thing and binds it: "<Subject 1> is \
a woman with ... . <Subject 1>'s appearance is given by <Picture 1> and <Picture 2>. \
<Audio 1> is the voice for <Subject 1> (S1)." Use only the references the user \
actually lists. If there are none, write N/A.

2. summary opens with a task-type prefix in square brackets, one of:
   [reference generation] an image/video/audio guides character, scene or style
   [keyframe completion]  an image is a concrete frame anchor
   [video continuation]   new content continues or extends a source video
   [video editing]        an existing source video is directly modified
   [audio reference]      only timbre/style is referenced, signal not copied
   [audio reuse]          the same audio signal is reused
   Use [video continuation] for extending a clip. [video editing] makes the model \
re-render the same action instead of continuing it.

3. retention_analysis gives every reference a marker. Visual: fully_preserved, \
partially_preserved, attribute_transfer, weak_reference. Audio: fully_copy, \
partially_copy, reference, weak_reference. For a voice saying NEW words use \
`reference`, never fully_copy.

4. detailed_description is the shot list in playback order, 350-500 words for a \
generation task. Open with one sentence of visual style. [Shot 1] opens untimed; \
later shots are "[Shot N] At MM:SS.mmm, ...". Never write prose time windows like \
"from 0:16 to 0:29" — they are paced far more loosely.

5. Dialogue and lyrics go in <d>[English] ... </d> tags and ONLY that text is \
spoken. The form is subject + speaker id + speech verb + tag: \
"<Subject 1> (S1) says warmly, <d>[English] You came all this way?</d>". Putting \
(S1) after </d> risks the id being read aloud.

6. overall_soundscape is diegetic ambience only — what exists in the space.

7. non_diegetic_music is score the characters CANNOT hear. An instrument a \
character is playing is diegetic and belongs in detailed_description. If in doubt \
write N/A. Never repeat dialogue there.

8. Describe composition, lighting, wardrobe and action as facts. Do not describe \
plot, motivation or backstory.

9. State camera behaviour positively: "a single continuous take from one locked-off \
tripod; the camera holds the same framing throughout". Do NOT write "no cuts" or \
"no camera movement" — naming a failure tends to produce it.

10. Camera moves are natural language with type, amplitude and speed: "a slow dolly \
toward her", not "camera: dolly-in".

Write nothing except the six sections."""


class H3RewriterBrief:
    """Build the system + user prompt for an LLM that writes H3 prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idea": ("STRING", {"multiline": True, "default":
                         "She waits by the window in a dim flat while it rains, then turns "
                         "when she hears the door.",
                         "tooltip": "One or two lines. What happens — the LLM turns it into "
                                    "a shot list."}),
                "seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0,
                                      "step": 0.5,
                              "tooltip": "The duration the BRIEF asks the writer "
                                         "to fill, so the shot markers it produces "
                                         "span this long. It does not set a render "
                                         "length — that comes out of H3 Rewriter "
                                         "Parse, which must be given the same "
                                         "number or the beats will not match the "
                                         "clip."}),
                "task_type": (["reference generation", "keyframe completion",
                               "video continuation", "video editing",
                               "audio reference", "audio reuse"],
                              {"default": "reference generation",
                               "tooltip": "What job the prompt is for. It opens "
                                          "the `summary` section the writer "
                                          "produces, which is how H3 is told what "
                                          "the references are FOR.\n\n"
                                          "reference generation — build a new clip "
                                          "from reference images.\n"
                                          "keyframe completion — a first and/or "
                                          "last frame are given.\n"
                                          "video continuation / editing — a source "
                                          "clip is being extended or altered.\n"
                                          "audio reference / reuse — the audio is "
                                          "the point."}),
                "shots": ("INT", {"default": 2, "min": 1, "max": 12,
                          "tooltip": "How many shots to ask for. One long take = 1."}),
                "dialogue": ("BOOLEAN", {"default": False,
                             "tooltip": "Ask for spoken lines in <d> tags."}),
            },
            "optional": {
                "subject_def_1": ("STRING", {"forceInput": True,
                                  "tooltip": "An EXISTING subject definition to "
                                             "hand the writer, so it describes "
                                             "your character instead of inventing "
                                             "one. Wire H3 Character's "
                                             "`subject_def`. Input-only: there is "
                                             "no widget, because hand-typing this "
                                             "is what H3 Scene Prompt is for."}),
                "retention_1": ("STRING", {"forceInput": True,
                                "tooltip": "Existing retention wording to preserve "
                                           "verbatim, from H3 Character's "
                                           "`retention`. Passing it in stops the "
                                           "writer rephrasing retention into the "
                                           "form that names <Picture N> as the "
                                           "retained thing, which makes the model "
                                           "render the anchor as a shot."}),
                "style_note": ("STRING", {"multiline": True, "default": "",
                               "tooltip": "A look to impose on the writer's first "
                                          "line — lens, grade, light, grain. "
                                          "Without one it picks its own, and a "
                                          "general-purpose model tends to pick "
                                          "'cinematic'. The LOOK only; staging and "
                                          "action belong in the idea."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Emit the system + user prompt for any LLM node, so it writes a "
                   "correct six-section H3 prompt. Feed the reply to H3 Rewriter Parse.")

    def go(self, idea, seconds, task_type, shots, dialogue,
           subject_def_1="", retention_1="", style_note=""):
        refs = subject_def_1.strip()
        user = [f"Write an H3 prompt for a {seconds:.1f} second clip in {shots} shot(s).",
                f"Use the task type [{task_type}].",
                f"The idea: {idea.strip()}"]
        if refs:
            user.append("Use EXACTLY these subject definitions verbatim as the "
                        "subject_definitions section, and no other references:\n" + refs)
        if retention_1.strip():
            user.append("Use EXACTLY this as the retention_analysis section:\n"
                        + retention_1.strip())
        if style_note.strip():
            user.append("Visual style: " + style_note.strip())
        user.append("Spoken dialogue in <d> tags is "
                    + ("REQUIRED." if dialogue else "NOT wanted; nobody speaks."))
        if shots == 1:
            user.append("One continuous shot: no additional [Shot N] markers after "
                        "[Shot 1].")
        return (SYSTEM, "\n\n".join(user))


class H3RewriterParse:
    """Normalise an LLM's reply into the six sections, and lint it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_output": ("STRING", {"multiline": True, "default": "",
                               "forceInput": False,
                               "tooltip": "The writer's REPLY, pasted or wired "
                                          "from whatever generated it. The six "
                                          "sections are found by their labels and "
                                          "normalised, so surrounding chatter and "
                                          "a code fence are tolerated — but a "
                                          "missing section cannot be invented, and "
                                          "`ok` goes false with the lint saying "
                                          "which. Nothing here reaches a model "
                                          "unchecked."}),
                "seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0,
                                      "step": 0.5,
                              "tooltip": "The duration to snap the `length` output "
                                         "to, on the 17n+5 grid. Set it to the "
                                         "SAME value the brief used: the beats in "
                                         "the text were written for that duration, "
                                         "and parsing at a different one leaves "
                                         "markers pointing past the end of the "
                                         "clip."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "length", "lint", "clean")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Strip an LLM's preamble and code fences, keep the six sections in the "
                   "right order, fill in any it omitted, and lint the result. Models "
                   "reliably add 'Here is your prompt:' and reorder sections.")

    def go(self, llm_output, seconds):
        text = (llm_output or "").strip()
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()

        found = {}
        for name in SECTIONS:
            m = re.search(rf"(?im)^\s*{name}\s*:\s*(.*?)(?=^\s*(?:{'|'.join(SECTIONS)})\s*:|\Z)",
                          text, re.S)
            if m:
                found[name] = m.group(1).strip()

        notes = []
        if not found:
            # no labels at all — treat the whole reply as the description rather than
            # failing, so a near-miss is still usable
            found["detailed_description"] = text
            notes.append("WARN parse: no section labels found; treated the whole reply "
                         "as detailed_description.")
        for name in SECTIONS:
            if name not in found:
                found[name] = "N/A"
                notes.append(f"WARN parse: '{name}' missing, filled with N/A.")

        prompt = "\n\n".join(f"{n}: {found[n]}" if n.startswith(("overall", "non_"))
                             else f"{n}:\n{found[n]}" for n in SECTIONS)
        n = max(5, round(seconds * 24))
        while n % 17 != 5:
            n += 1
        findings = lint(prompt)
        order = {"ERROR": 0, "WARN": 1, "INFO": 2}
        findings.sort(key=lambda f: order[f[0]])
        report = "\n".join(notes + [f"{s:5s} {r}: {m}" for s, r, m in findings]) \
            or "clean - no issues found"
        errs = sum(1 for f in findings if f[0] == "ERROR")
        return {"ui": {"h3lint": [report]},
                "result": (prompt, n, report, errs == 0)}


NODE_CLASS_MAPPINGS = {"H3RewriterBrief": H3RewriterBrief,
                       "H3RewriterParse": H3RewriterParse}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RewriterBrief": "H3 Rewriter Brief (to LLM)",
                              "H3RewriterParse": "H3 Rewriter Parse (from LLM)"}
