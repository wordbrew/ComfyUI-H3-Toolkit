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
    CLAUSE  what changes this link: the subject's STATE and action.
            Self-contained -- "her forearms are streaked with clay", not "her
            forearms are now streaked with clay".
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
                         "Realistic video. A woman in her thirties with dark "
                         "shoulder-length hair and a rust knit jumper works at a long "
                         "wooden bench in a bright ceramics studio. Whitewashed walls, "
                         "tall windows on the left, shelves of unglazed pots behind her.",
                         "tooltip": "Subject, setting, lighting. IDENTICAL on every link — "
                                    "never 'the same room', which assumes a memory the "
                                    "model does not have."}),
                "beats": ("STRING", {"multiline": True, "default":
                          "Her sleeves are pushed to the elbow and her hands are "
                          "clean. She centres a ball of clay on the wheel, both palms "
                          "braced against it.\n"
                          "Her forearms are streaked with wet clay. She opens the "
                          "centre of the spinning clay with both thumbs.\n"
                          "Her forearms and the front of her jumper are streaked with "
                          "drying grey clay. She draws the wall of the pot upward "
                          "between finger and thumb.\n"
                          "Her hands are washed and her jumper is dusted pale with dried "
                          "clay. She lifts the finished pot from the wheel with both "
                          "hands and sets it on the bench.",
                          "tooltip": "ONE LINE PER LINK. Each must be self-contained: "
                                     "state it as a FACT ('her forearms are "
                                     "streaked with clay'), not as a change "
                                     "('her forearms are now streaked'). Each "
                                     "link is rendered on its own and has no "
                                     "memory of the one before it."}),
                "tail": ("STRING", {"multiline": True, "default":
                         "A single continuous take from one locked-off tripod at bench "
                         "height; the camera holds the same framing of her hands and "
                         "upper body for the entire duration and the take runs unbroken "
                         "from the first frame to the last.",
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
                                     "tooltip": "IGNORED when `chunk_frames` is "
                                                "wired — under chunking the plan "
                                                "sets each chunk's length and the "
                                                "tail is never the same as the "
                                                "rest. For the manual chain "
                                                "workflow: 15.08s (362 frames) is "
                                                "the top of the documented trained "
                                                "range for VIDEO."}),
                "seed": ("INT", {"default": 2024, "min": 0, "max": 0xffffffffffffffff,
                         "tooltip": "Passed through unchanged. Use the SAME seed on every "
                                    "link — varying it per link measurably broke "
                                    "continuity. Wire the `seed` output to the "
                                    "sampler's noise_seed under chunking too — "
                                    "one seed for every chunk is the same rule, "
                                    "and it keeps the seed next to the prompt."}),
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
                # APPENDED, and they stay appended. widgets_values is positional.
                "chunk_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                                 "tooltip": "Wire H3 Chunk Open's `length`. Under "
                                            "chunking the chunk decides the "
                                            "length, so this REPLACES "
                                            "seconds_per_link — which is then "
                                            "ignored, and the plan stops quoting "
                                            "it. Leave at 0 for the manual chain "
                                            "workflow, where seconds_per_link is "
                                            "still the real setting."}),
                # APPENDED. Both sections used to be hardcoded "N/A", which is
                # not a neutral default -- it is the prompt actively saying
                # there is no ambience and no music, so a graph that decoded the
                # audio still got near-silence and the cause was invisible.
                "soundscape": ("STRING", {"multiline": True, "default": "",
                               "tooltip": "Diegetic sound: the room, the "
                                          "surfaces, what is physically making "
                                          "noise. IDENTICAL on every chunk, like "
                                          "head and tail — it describes the "
                                          "place, not the moment. Empty means "
                                          "N/A, which asks for silence."}),
                "music": ("STRING", {"multiline": True, "default": "",
                          "tooltip": "Non-diegetic score. Empty means N/A. "
                                     "Generated music compounds down a chain "
                                     "('photocopy of a photocopy'); an external "
                                     "track sliced per chunk into ref_audio does "
                                     "not, and beat it in testing."}),
                "chunk_plan": ("H3_CHUNK_PLAN", {"tooltip": "Optional, from H3 "
                               "Chunk Plan. Only so the plan text can line each "
                               "clause up against the chunk it will actually be "
                               "rendered on."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "length", "seed", "link_count", "plan", "lint",
                    "clause")
    # `length` and `seed` are for the MANUAL CHAIN workflow. Under chunking the
    # length belongs to H3 Chunk Open (the tail chunk differs from the rest) and
    # the seed to the shared RandomNoise. Wiring `length` from here is how a
    # chunk ends up asking for 56 generated frames against 39 source ones.
    OUTPUT_TOOLTIPS = ("The finished prompt for this link or chunk.",
                       "MANUAL CHAIN ONLY — under chunking use H3 Chunk Open's "
                       "`length`, which knows this chunk's own run.",
                       "One seed for every link or chunk — varying it per link "
                       "measurably broke continuity. Wire it to the sampler's "
                       "noise_seed, or leave the sampler to hold its own.",
                       "How many clauses `beats` holds.",
                       "The clause-to-chunk map. Read it before you queue.",
                       "Errors and warnings. Wire this one if you wire only one.",
                       "Just this link's clause, for previewing.")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/prompt"
    DESCRIPTION = ("Build one link's prompt for a chained long-form take: HEAD + CLAUSE + "
                   "TAIL, byte-identical except the clause. Enforces the rules the wave "
                   "log settled — self-contained prompts, no relative phrasing, a "
                   "continuity clause, and one seed across links.")

    def build(self, head, beats, tail, link_index, seconds_per_link, seed,
              subject_def_1="", retention_1="", task_type="keyframe completion",
              expected_count=0, soundscape="", music="", chunk_frames=0,
              chunk_plan=None):
        # A chunk's length comes from the plan, not from a seconds widget: the
        # tail chunk is never the same length as the rest, so a single duration
        # cannot describe the run. When it is wired, it wins outright.
        chunked = int(chunk_frames or 0) > 0
        frames = int(chunk_frames) if chunked else align_frames(seconds_per_link)
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
        # Line the clauses up against the chunks they will be RENDERED on. A
        # clause list on its own does not tell you whether clause 3 describes the
        # part of the clip you think it does.
        chunks = (chunk_plan or {}).get("chunks") or []

        body = f"{h} {clause} {t}".strip()
        defs, rets = subject_def_1.strip(), retention_1.strip()
        if defs:
            prompt = ("subject_definitions:\n" + defs + "\n\n"
                      "summary:\n"
                      f"[{task_type}] A single continuous take, link {idx + 1} of "
                      f"{len(clauses)}, using the listed references for identity.\n\n"
                      "retention_analysis:\n" + (rets or "N/A") + "\n\n"
                      "detailed_description:\n" + body + "\n\n"
                      "overall_soundscape: " + (soundscape.strip() or "N/A") +
                      "\n\nnon_diegetic_music: " + (music.strip() or "N/A"))
        else:
            prompt = body

        # --- guidance, checked per part so the message points at the right box
        notes = []
        if defs and not soundscape.strip() and not music.strip():
            notes.append("WARN audio: overall_soundscape and non_diegetic_music "
                         "are both N/A, which asks for SILENCE. If you are "
                         "decoding the audio, describe the room here.")
        if _mismatch:
            notes.append(_mismatch)
        if chunks and len(chunks) != len(clauses):
            notes.append(f"ERROR count: the wired plan has {len(chunks)} chunk(s) "
                         f"but `beats` has {len(clauses)} clause(s). Extra chunks "
                         f"REPEAT the last clause; extra clauses are never used.")
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

        rows = []
        for i, c in enumerate(clauses):
            here = "  <-- emitting" if i == idx else ""
            span = ""
            if i < len(chunks):
                k = chunks[i]
                span = (f"  frames {k['keep_from']}-{k['end']} ({k['run']}f"
                        f"{'' if k['both_clocks'] else ', OFF audio grid'})")
            rows.append(f"  clause {i + 1}/{len(clauses)}{span}{here}\n"
                        f"      {c[:88]}")

        if chunked:
            gen = sum(k["run"] for k in chunks) if chunks else frames * len(clauses)
            head_line = (f"{len(clauses)} clause(s) against "
                         f"{len(chunks) or '?'} chunk(s)  |  this chunk is "
                         f"{frames} frames ({frames / FPS:.2f}s)")
            if chunks:
                kept = sum(k["end"] - k["keep_from"] for k in chunks)
                head_line += (f"\n{kept} frames kept ({kept / FPS:.1f}s), "
                              f"{gen} generated")
            head_line += ("\nlength and seed come from the chunk runner — "
                          "seconds_per_link is ignored")
        else:
            head_line = (f"{len(clauses)} links x {dur:.2f}s = "
                         f"{len(clauses) * dur:.1f}s total  |  {frames} "
                         f"frames/link  |  seed {seed} on every link")
        plan_text = head_line + "\n" + "\n".join(rows)
        report = "\n".join(notes) if notes else "clean - no issues found"
        return {"ui": {"h3plan": [plan_text], "h3lint": [report]},
                "result": (prompt, frames, seed, len(clauses), plan_text, report,
                           clause)}


NODE_CLASS_MAPPINGS = {"H3LongFormLinks": H3LongFormLinks}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongFormLinks": "H3 Long-form Links"}
