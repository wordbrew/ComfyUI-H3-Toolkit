"""H3 Long-form Links — one prompt per link of a chained take.

Twelve waves of engine work (~/projects/h3/docs/long-form-waves.md) reduce to a
small number of rules about how the prompts must be written. This writes them.

THE RULE THAT MATTERS MOST: every link is generated INDEPENDENTLY. The model
carries no memory of the previous prompt, so a link saying "continuing the same
shot" has nothing to resolve that against and re-invents the scene — which reads
as a hard cut a second or two in. That single mistake is what made waves 1-5
incoherent, and fixing it is what made wave 6 work.

So each link is HEAD + CLAUSE + TAIL, where HEAD and TAIL are BYTE-IDENTICAL
across every link and only the clause changes:

    HEAD    who and where: subject, setting, lighting. Never "the same room".
    CLAUSE  what changes this link: wardrobe state and action. Self-contained —
            "She is topless in a black lace thong", not "she is now topless".
    TAIL    camera and continuity, stated positively: "a single continuous take,
            the camera holds the same framing, the take runs unbroken", plus the
            audio line. NOT "no cuts / no camera movement" — naming the failure
            produces it (090 cut twice; 091 without those words was clean at two
            seeds, one of them the seed that had cut).

WHAT THIS NODE DOES NOT DO, AND WHY
  It does not chain the frames — that is H3 Chain Frame plus a keyframe input on
  the conditioning node. It does not lock the audio — that is H3 Audio Lock, fed
  a slice of one long soundtrack. This is only the prompt side, because the prompt
  side is where the non-obvious knowledge lives.

SETTINGS THE WAVES SETTLED, surfaced here as guidance rather than silently applied:
  - SAME SEED on every link. Different seeds per link measurably broke continuity.
    The `seed` input is passed straight through so the rule is visible in the graph.
  - 640x1120 worked; 768x1152 produced cuts in chained clips with everything else
    held constant.
  - NO latent pin and NO tail video reference. The pin cuts at its boundary; a
    reference acts as a destination and the clip converges onto it (ping-pong).
    Both were chased for nine waves and neither is separable from its side effect.
  - A single-frame keyframe from the previous link's last frame is the mechanism
    that works. It cannot act as a destination because it is bound to frame 0.

Known residual defect, so you are not surprised by it: a still frame carries POSE
but not VELOCITY, so motion sometimes resumes at the wrong speed or direction.
Every mechanism that could carry velocity is a reference, and references converge.
"""

import re

from .prompt_lint import lint

FPS = 24
RELATIVE_HINT = re.compile(
    r"\b(continuing|continues from|as before|the same (?:shot|scene|room|setting)|"
    r"previously|still in the|picking up|now (?:topless|nude|wearing))\b", re.I)


def align_frames(seconds):
    n = max(5, round(seconds * FPS))
    while n % 17 != 5:
        n += 1
    return n


def parse_beats(text):
    """One clause per non-empty line. Blank lines and '#' comments ignored."""
    out = []
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


class H3LongFormLinks:
    """Emit the prompt for one link of a chained long-form take."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head": ("STRING", {"multiline": True, "default":
                         "Realistic video. A woman with curly copper-red hair, freckled "
                         "fair skin and a curvy figure dances at a floor-to-ceiling chrome "
                         "pole in a dim club room. Dark walls, warm amber light spilling "
                         "from the left, faint haze in the air.",
                         "tooltip": "Subject, setting, lighting. IDENTICAL on every link — "
                                    "never 'the same room', which assumes a memory the "
                                    "model does not have."}),
                "beats": ("STRING", {"multiline": True, "default":
                          "She wears a black lace bra and matching thong. She circles the "
                          "pole, hooks one hand high on it and spins slowly, hips rolling "
                          "to the beat.\n"
                          "She wears a black lace bra and matching thong. She dances "
                          "against the pole, reaches back, unhooks the bra and slides it "
                          "off, then keeps dancing topless.\n"
                          "She is topless in a black lace thong. She dances at the pole, "
                          "one hand sliding down the chrome, hips rolling to the beat.\n"
                          "She is fully nude. She dances slowly around the pole, leaning "
                          "back with one arm extended, hips rolling to the beat.",
                          "tooltip": "ONE LINE PER LINK. Each must be self-contained: "
                                     "state the wardrobe as a fact ('She is topless'), not "
                                     "as a change ('she is now topless')."}),
                "tail": ("STRING", {"multiline": True, "default":
                         "A single continuous take from one locked-off tripod at waist "
                         "height; the camera holds the same framing of her full body for "
                         "the entire duration and the take runs unbroken from the first "
                         "frame to the last. Audio: a slow sultry R&B track, steady "
                         "bassline, quiet room reverb, no speech.",
                         "tooltip": "Camera + continuity + audio. IDENTICAL on every link. "
                                    "Say what the camera DOES. Do NOT write 'no cuts / no "
                                    "camera movement' — naming the failure produces it "
                                    "(090 cut twice; 091 without those words was clean)."}),
                "link_index": ("INT", {"default": 0, "min": 0, "max": 63,
                               "tooltip": "Which link to emit, 0-based. Render one link per "
                                          "queue run, feeding the previous link's last "
                                          "frame in as the keyframe."}),
                "seconds_per_link": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 15.5,
                                     "step": 0.5,
                                     "tooltip": "15.08s (362 frames) is the top of the "
                                                "documented trained range for VIDEO."}),
                "seed": ("INT", {"default": 2024, "min": 0, "max": 0xffffffffffffffff,
                         "tooltip": "Passed through unchanged. Use the SAME seed on every "
                                    "link — varying it per link measurably broke "
                                    "continuity."}),
            },
            "optional": {
                # typed OR wired, same as H3 Scene Prompt
                "subject_def_1": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Wire H3 Character, or type the wording here. "
                                             "Leave empty for a plain single-string prompt "
                                             "with no reference sections."}),
                "retention_1": ("STRING", {"multiline": True, "default": ""}),
                "expected_count": ("INT", {"default": 0, "min": 0, "max": 4096,
                                   "tooltip": "Wire H3 Chunk Open's chunk_count "
                                              "here. If it does not match the "
                                              "number of clauses, the lint says "
                                              "so — otherwise a mismatch just "
                                              "CLAMPS and the last clause repeats "
                                              "silently for every extra chunk. "
                                              "0 = no check."}),
                "task_type": (["reference generation", "keyframe completion"],
                              {"default": "keyframe completion"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "length", "seed", "link_count", "plan", "lint")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Build one link's prompt for a chained long-form take: HEAD + CLAUSE + "
                   "TAIL, byte-identical except the clause. Enforces the rules the wave "
                   "log settled — self-contained prompts, no relative phrasing, a "
                   "continuity clause, and one seed across links.")

    def build(self, head, beats, tail, link_index, seconds_per_link, seed,
              subject_def_1="", retention_1="", task_type="keyframe completion",
              expected_count=0):
        frames = align_frames(seconds_per_link)
        dur = frames / FPS
        clauses = parse_beats(beats)
        if not clauses:
            raise ValueError("`beats` is empty — one line per link.")
        want = int(expected_count or 0)
        if want and want != len(clauses):
            over = "more chunks than clauses" if want > len(clauses) else \
                   "more clauses than chunks"
            _mismatch = (f"ERROR count: {len(clauses)} clause(s) for {want} chunk(s) "
                         f"— {over}. Extra chunks REPEAT the last clause, and "
                         f"extra clauses are never used. Add or remove beats so "
                         f"the counts match.")
        else:
            _mismatch = None

        idx = max(0, min(link_index, len(clauses) - 1))
        h, t = head.strip(), tail.strip()
        clause = clauses[idx]

        body = f"{h} {clause} {t}".strip()
        defs, rets = subject_def_1.strip(), retention_1.strip()
        if defs:
            prompt = ("subject_definitions:\n" + defs + "\n\n"
                      "summary:\n"
                      f"[{task_type}] A single continuous take, link {idx + 1} of "
                      f"{len(clauses)}, using the listed references for identity.\n\n"
                      "retention_analysis:\n" + (rets or "N/A") + "\n\n"
                      "detailed_description:\n" + body + "\n\n"
                      "overall_soundscape: N/A\n\nnon_diegetic_music: N/A")
        else:
            prompt = body

        # --- guidance, checked per part so the message points at the right box
        notes = []
        if _mismatch:
            notes.append(_mismatch)
        for label, txt in (("head", h), ("tail", t)):
            hits = RELATIVE_HINT.findall(txt)
            if hits:
                notes.append(f"ERROR {label}: relative phrasing {sorted(set(hits))[:3]} — "
                             f"HEAD and TAIL are repeated verbatim on every link, so "
                             f"anything relative is wrong on all of them.")
        for i, c in enumerate(clauses):
            hits = RELATIVE_HINT.findall(c)
            if hits:
                notes.append(f"ERROR beat {i + 1}: relative phrasing {sorted(set(hits))[:2]}"
                             f" — state the wardrobe as a fact, not as a change.")
        if not re.search(r"\b(single|one) (?:uninterrupted |unbroken )?continuous take\b"
                         r"|\bruns unbroken\b|\bholds? the same framing\b", t, re.I):
            notes.append("WARN tail: no positive continuity clause. State what the camera "
                         "DOES — 'a single continuous take', 'the camera holds the same "
                         "framing', 'the take runs unbroken'.")
        if len({len(c.split()) for c in clauses}) and max(len(c.split()) for c in clauses) < 8:
            notes.append("WARN beats: very short clauses. Each must describe wardrobe AND "
                         "action on its own; the model sees nothing else that changes.")
        notes += [f"{s} {r}: {m}" for s, r, m in lint(prompt, long_form=True)
                  if r != "longform/relative"]      # already reported per-part above

        plan = [f"link {i + 1}/{len(clauses)}{'  <-- emitting' if i == idx else ''}: "
                f"{c[:72]}" for i, c in enumerate(clauses)]
        plan_text = (f"{len(clauses)} links x {dur:.2f}s = {len(clauses) * dur:.1f}s total"
                     f"  |  {frames} frames/link  |  seed {seed} on every link\n"
                     + "\n".join(plan))
        report = "\n".join(notes) if notes else "clean - no issues found"
        return {"ui": {"h3plan": [plan_text], "h3lint": [report]},
                "result": (prompt, frames, seed, len(clauses), plan_text, report)}


NODE_CLASS_MAPPINGS = {"H3LongFormLinks": H3LongFormLinks}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongFormLinks": "H3 Long-form Links"}
