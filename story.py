"""Design a take in SHOTS and LINES, and let the frame arithmetic follow.

Both nodes here exist because the same sums were being done by hand, on paper,
every time a length or a chunk size changed -- and got wrong.

  `H3Shotlist`   you know the piece is four shots of eight to twelve seconds.
                 Turning that into total_frames, cut points and the right NUMBER
                 of prompt clauses is arithmetic against the 17n+5 grid, the
                 audio grid and the chunk stride, and it changes completely when
                 you touch chunk_frames.

  `H3Dialogue`   a line's timestamp is relative to its OWN chunk, and the first
                 `pin` frames of every chunk after the first are reproduced from
                 the previous one, so nothing new can be spoken there. A line
                 scheduled at 00:01.500 against a 39-frame pin (1.625s) is simply
                 never said, and you find out nine minutes into a render.
                 Measured 2026-08-28: that exact line vanished.

Neither node generates anything. They produce text you can read and edit before
you spend a render on it.
"""

import logging
import re

from .chunkplan import legal_run, plan as build_plan, snap_context
from .chunkplan import describe as describe_plan

FPS = 24.0
CATEGORY = "MiniMax H3/long-form"

_VOWELS = re.compile(r"[aeiouy]+", re.I)
_SPEAKER = re.compile(r"^\s*[sS](\d+)\s*$")


def stamp(seconds):
    """MM:SS.mmm — the form H3's prompt format expects for a dialogue anchor.

    Rounded to a tenth. The underlying number comes from a syllable estimate, so
    three decimal places would be claiming a precision that is not there.
    """
    t = round(float(seconds), 1)
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def syllables(text):
    """Rough syllable count, for estimating how long a line takes to say.

    Vowel GROUPS, minus a silent trailing e, floor of one per word. It is not
    linguistics -- it only has to rank lines by length well enough to space
    them, and words-per-second is measurably worse at that because "I know" and
    "unquestionably" are both one or two words.
    """
    n = 0
    for word in re.findall(r"[A-Za-z']+", text):
        c = len(_VOWELS.findall(word))
        if word.lower().endswith("e") and c > 1:
            c -= 1
        n += max(1, c)
    return n


def parse_rows(text, sep="|"):
    """Non-empty, non-comment lines split on `sep`, fields stripped."""
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([f.strip() for f in line.split(sep)])
    return rows


def chunk_windows(chunks, lead_in=0.075, tail_margin=0.6):
    """Per chunk: (local speech window, joined-time offset), in seconds.

    Two separate facts, and both bite.

    The FLOOR is the pin. The first `pin` frames of a chunk are reproduced from
    the previous one under a denoise mask of 0, so the model cannot put new
    speech there -- it is not a soft preference, those positions are stamped
    back at every denoise step.

    The OFFSET is not the chunk's start frame. The join drops everything before
    `keep_from`, so a chunk's local t=0 does not appear in the output at all. A
    line's real time is `local + (keep_from - start_of_kept_region)`, which is
    the cumulative kept length before it, minus the pin it sits behind.
    """
    out, pos = [], 0
    for c in chunks:
        pin_s = int(c.get("pin", 0)) / FPS
        run_s = int(c["run"]) / FPS
        lo = (pin_s + lead_in) if c.get("pin") else min(0.6, run_s * 0.1)
        hi = max(lo, run_s - tail_margin)
        # joined time of this chunk's local t: kept frames before it, less the
        # part of this chunk that is dropped
        offset = pos / FPS - (int(c["keep_from"]) - int(c["start"])) / FPS
        out.append({"lo": lo, "hi": hi, "offset": offset,
                    "run_s": run_s, "pin_s": pin_s})
        pos += int(c["end"]) - int(c["keep_from"])
    return out


class H3Shotlist:
    """Write the shots; get the frame counts, the cuts and the clause scaffold.

    WHY THIS IS NOT JUST DIVISION
      A shot's length has to be a legal run (17n+5) or the planner corrects it
      later and the number you designed to is not the number rendered. Shots
      then become CUTS, and a chunk that starts a shot carries pin 0 -- no
      inherited prefix. That matters beyond tidiness: local contrast on skin was
      measured accumulating +12.4%, +6.9%, +4.5% per chunk down a chained take,
      with the increment landing at each seam. A cut is where that carry stops.

    WHAT IT DELIBERATELY DOES NOT DO
      It does not write your prompt. The `beats` output is a SCAFFOLD -- one
      line per chunk, prefilled with that chunk's shot so the count is right and
      the framing is in front of you. The words are yours.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shots": ("STRING", {"multiline": True, "default":
                      "10 | locked wide, the boardwalk running away behind them "
                      "| they walk toward the camera, talking\n"
                      "10 | medium two-shot at chest height "
                      "| she turns to the rail and looks out at the water\n"
                      "10 | closer, the sea behind her "
                      "| she looks back into the lens and stops",
                      "tooltip": "One shot per line: SECONDS | framing | action. "
                                 "Blank lines and # comments are ignored. Each "
                                 "shot is snapped to a legal run, so what you "
                                 "type is what gets rendered."}),
            "chunk_frames": ("INT", {"default": 243, "min": 5, "max": 3600,
                             "step": 17,
                             "tooltip": "Same value you will put on H3 Chunk "
                                        "Plan. Shots longer than this are split "
                                        "into several chunks, and each split "
                                        "needs its own clause."}),
            "context": (["39", "22", "5", "1", "0"], {"default": "39",
                         "tooltip": "Same value as the plan's. It changes the "
                                    "chunk count, so the clause count depends "
                                    "on it."}),
        }}

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("total_frames", "cut_frames", "beats", "info")
    OUTPUT_TOOLTIPS = ("Wire to H3 Chunk Plan's total_frames.",
                       "Wire to H3 Chunk Plan's cut_frames. These are where a "
                       "new shot begins, and where the chain resets.",
                       "A clause scaffold with the right NUMBER of lines, one "
                       "per chunk, prefilled with its shot. Edit the words.",
                       "The shot table: what each one rounded to, and how many "
                       "chunks it costs.")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Turn a shot list into total_frames, cut points and a clause "
                   "scaffold with the right number of lines.")

    def go(self, shots, chunk_frames, context):
        rows = parse_rows(shots)
        if not rows:
            msg = "H3 SHOTLIST: no shots — one per line, SECONDS | framing | action"
            return {"ui": {"h3char": [msg]}, "result": (0, "", "", msg)}

        ctx = snap_context(int(context))
        size = legal_run(max(5, int(chunk_frames)), "up")

        parsed, cuts, pos, notes = [], [], 0, []
        for i, r in enumerate(rows, 1):
            try:
                secs = float(r[0])
            except (ValueError, IndexError):
                raise ValueError(
                    f"H3 Shotlist: shot {i} does not start with a duration in "
                    f"seconds — got {r[0]!r}. Format is SECONDS | framing | action.")
            framing = r[1] if len(r) > 1 else ""
            action = r[2] if len(r) > 2 else ""
            asked = max(5, int(round(secs * FPS)))
            frames = legal_run(asked, "up")
            if frames != asked:
                notes.append(f"shot {i}: {secs:g}s is {asked} frames, not a legal "
                             f"run — using {frames} ({frames / FPS:.2f}s)")
            if pos:
                cuts.append(pos)
            parsed.append({"i": i, "start": pos, "frames": frames,
                           "framing": framing, "action": action})
            pos += frames

        total = pos
        chunks, info = build_plan(total, size, "scene", cuts=cuts, context=ctx,
                                  grow_tail=True, generated_audio=True)

        # which shot does each chunk belong to, so the scaffold can name it
        lines = []
        for c in chunks:
            shot = next((s for s in reversed(parsed) if c["keep_from"] >= s["start"]),
                        parsed[0])
            part = f" ({c['part'][0]} of {c['part'][1]})" if c.get("part") else ""
            lines.append(f"# shot {shot['i']}{part} — {shot['framing']}\n"
                         f"{shot['action']}")

        L = [f"H3 SHOTLIST — {len(parsed)} shot(s), {total} frames "
             f"({total / FPS:.2f}s), {len(chunks)} chunk(s)"]
        L.append(f"{'shot':>5}{'frames':>14}{'seconds':>10}   framing")
        for s in parsed:
            n_here = sum(1 for c in chunks
                         if s["start"] <= c["keep_from"] < s["start"] + s["frames"])
            span = f"{s['start']}-{s['start'] + s['frames']}"
            L.append(f"{s['i']:>5}{span:>14}"
                     f"{s['frames'] / FPS:>9.2f}s   {s['framing'][:44]}"
                     f"   [{n_here} chunk(s)]")
        L.append("")
        L.append(f"  cut_frames: {', '.join(str(c) for c in cuts) or '(none)'}")
        L.append(f"  a chunk starting a shot carries pin 0 — the chain resets "
                 f"there, and so does the contrast carried with it")
        L.append("")
        L.append(describe_plan(chunks, info))
        for n in notes:
            L.append(f"  NOTE {n}")

        text = "\n".join(L)
        logging.info("H3Shotlist: %s", L[0])
        return {"ui": {"h3char": [text]},
                "result": (total, ", ".join(str(c) for c in cuts),
                           "\n".join(lines), text)}


class H3Dialogue:
    """Lay lines out across the chunks, at times they can actually be spoken.

    THE FLOOR THAT IS NOT OBVIOUS
      Timestamps are relative to each chunk's own start, and the first `pin`
      frames of every chunk after the first are REPRODUCED from the previous
      chunk under a denoise mask of 0. Nothing new happens there. A line at
      00:01.500 behind a 39-frame pin (1.625s) is inside that region and is
      never said -- measured 2026-08-28, and invisible until you listen to a
      finished render.

    THE CEILING
      A line that runs past the end of its chunk is cut off mid-word, and the
      join then cuts to the next chunk. `tail_margin` is the room left for it to
      finish.

    WHAT IT COSTS AT A SEAM
      A gap of roughly `pin/24` seconds at every chunk boundary is not a choice,
      it is the pin. Two seams is two of them. The node reports the joined-time
      schedule so you can see whether the pauses land between exchanges, where
      they read as a breath, or in the middle of one, where they read as a fault.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lines": ("STRING", {"multiline": True, "default":
                      "S1 | says lightly | He texted me at two in the morning.\n"
                      "S2 | asks | And you answered?\n"
                      "S1 | replies | I let him wait until noon.\n"
                      "S2 | says | That is mean.",
                      "tooltip": "One line per spoken line: SPEAKER | verb | "
                                 "text, or SPEAKER | text. A blank line forces "
                                 "a break to the next chunk. # comments are "
                                 "ignored."}),
            "chunk_plan": ("H3_CHUNK_PLAN", {"tooltip": "From H3 Chunk Plan. "
                           "This is where the pin, the run lengths and the join "
                           "positions come from — without it the timings are "
                           "guesses."}),
        }, "optional": {
            "actions": ("STRING", {"multiline": True, "default": "",
                        "tooltip": "One NON-SPEECH line per chunk, appended to "
                                   "that chunk's clause — what the camera or the "
                                   "world does while they talk. Blank lines "
                                   "allowed; short lists just stop."}),
            "syllables_per_second": ("FLOAT", {"default": 4.3, "min": 1.0,
                                     "max": 10.0, "step": 0.1,
                                     "tooltip": "Conversational English is about "
                                                "4.3. Lower it if lines are "
                                                "being cut off; raise it if the "
                                                "gaps are too long."}),
            "gap": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Seconds between one line ending and the next "
                               "starting, inside a chunk."}),
            "tail_margin": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0,
                            "step": 0.1,
                            "tooltip": "Room left at the end of a chunk for the "
                                       "last line to finish."}),
            "speaker_map": ("STRING", {"multiline": False, "default": "",
                            "tooltip": "When a speaker is not the subject of the "
                                       "same number: S2=<Subject 3>, comma "
                                       "separated. Default maps S1 to "
                                       "<Subject 1>."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("beats", "timeline", "lint")
    OUTPUT_TOOLTIPS = ("One clause per chunk, ready for H3 Long-form Links.",
                       "The joined-time schedule — when each line is actually "
                       "heard, and the gaps between them.",
                       "Lines that would not fit, chunks left silent, and any "
                       "clause that overran.")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Place spoken lines across the chunks at times they can "
                   "actually be spoken, and report when they are heard.")

    def go(self, lines, chunk_plan, actions="", syllables_per_second=4.3,
           gap=0.4, tail_margin=0.6, speaker_map=""):
        chunks = (chunk_plan or {}).get("chunks") or []
        if not chunks:
            msg = "H3 DIALOGUE: no plan wired — nothing to lay lines out against."
            return {"ui": {"h3char": [msg]}, "result": ("", "", msg)}

        who = {}
        for pair in speaker_map.replace(";", ",").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                who[k.strip().upper()] = v.strip()

        # a blank line is a deliberate break to the next chunk
        groups, cur = [], []
        for raw in (lines or "").splitlines():
            line = raw.strip()
            if line.startswith("#"):
                continue
            if not line:
                if cur:
                    groups.append(cur); cur = []
                continue
            f = [x.strip() for x in line.split("|")]
            spk = f[0]
            m = _SPEAKER.match(spk)
            if not m:
                raise ValueError(
                    f"H3 Dialogue: {spk!r} is not a speaker tag. Use S1, S2 ... "
                    f"as the first field.")
            verb = f[1] if len(f) > 2 else "says"
            text = f[-1] if len(f) > 1 else ""
            if not text:
                raise ValueError(f"H3 Dialogue: {line!r} has no spoken text.")
            cur.append({"n": int(m.group(1)), "verb": verb, "text": text,
                        "dur": syllables(text) / max(0.5, float(syllables_per_second))})
        if cur:
            groups.append(cur)

        win = chunk_windows(chunks, tail_margin=float(tail_margin))
        acts = [a for a in (actions or "").splitlines()]

        # pack: honour a forced break, otherwise fill each chunk to its ceiling
        queue = [ln for g in groups for ln in ([None] + g)][1:]   # None = break
        placed = [[] for _ in chunks]
        ci, t, notes = 0, None, []
        for item in queue:
            if item is None:
                ci += 1
                t = None
                continue
            while ci < len(chunks):
                w = win[ci]
                start = w["lo"] if t is None else t + float(gap)
                if start + item["dur"] <= w["hi"] + 1e-6:
                    placed[ci].append(dict(item, at=start))
                    t = start + item["dur"]
                    break
                ci += 1
                t = None
            else:
                notes.append(f"ERROR overflow: {item['text'][:44]!r} did not fit "
                             f"in any remaining chunk. Add a chunk, shorten the "
                             f"line, or raise syllables_per_second.")
                break

        beats, rows = [], []
        for i, (c, w, items) in enumerate(zip(chunks, win, placed)):
            parts = []
            for it in items:
                subj = who.get(f"S{it['n']}", f"<Subject {it['n']}>")
                parts.append((stamp(it["at"]),
                              f"{subj} (S{it['n']}) {it['verb']}, "
                              f"<d>[English] {it['text']}</d>"))
                rows.append((w["offset"] + it["at"],
                             w["offset"] + it["at"] + it["dur"], i + 1, it))
            # "At ... at ... and at ..." — the shape the wave-log prompts use.
            # Capitalising every anchor reads as a list of stage directions
            # rather than a described take.
            said = []
            for k, (ts, body) in enumerate(parts):
                lead = "At" if k == 0 else ("and at" if k == len(parts) - 1
                                            else "at")
                said.append(f"{lead} {ts} {body}")
            act = acts[i].strip() if i < len(acts) else ""
            clause = " ".join(said) if parts else ""
            if act:
                clause = (clause + " while " + act) if clause else act
            if not clause:
                clause = "# EMPTY — this chunk has no line and no action"
                notes.append(f"WARN chunk {i + 1} has nothing in it. Every chunk "
                             f"needs a clause or the last one repeats there.")
            beats.append(clause)

        L = [f"H3 DIALOGUE — {sum(len(p) for p in placed)} line(s) across "
             f"{len(chunks)} chunk(s)"]
        L.append(f"{'heard':>16}{'chunk':>7}   line")
        prev_end = None
        for a, b, ck, it in sorted(rows):
            g = "" if prev_end is None else f"  (+{a - prev_end:.2f}s)"
            L.append(f"{f'{a:6.2f}-{b:5.2f}s':>16}{ck:>7}   "
                     f"S{it['n']}: {it['text'][:52]}{g}")
            prev_end = b
        L.append("")
        for i, w in enumerate(win, 1):
            L.append(f"  chunk {i}: speech window {w['lo']:.3f}–{w['hi']:.2f}s "
                     f"local, {w['pin_s']:.3f}s held by the pin, "
                     f"joined offset {w['offset']:+.2f}s")

        gaps = []
        srt = sorted(rows)
        for (a0, b0, c0, _), (a1, _, c1, _) in zip(srt, srt[1:]):
            if c1 != c0:
                gaps.append(f"gap {a1 - b0:.2f}s of silence between chunk {c0} "
                            f"and chunk {c1}, at {b0:.2f}s")
        report = "\n".join(notes + gaps) if (notes or gaps) else \
            "clean — every line fits, no silence across a seam"
        text = "\n".join(L)
        logging.info("H3Dialogue: %s", L[0])
        return {"ui": {"h3char": [text], "h3lint": [report]},
                "result": ("\n".join(beats), text, report)}


NODE_CLASS_MAPPINGS = {"H3Shotlist": H3Shotlist, "H3Dialogue": H3Dialogue}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Shotlist": "H3 Shot List",
                              "H3Dialogue": "H3 Dialogue (timed to chunks)"}
