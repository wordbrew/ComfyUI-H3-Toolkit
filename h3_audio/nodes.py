"""H3 Audio Prompt — build MiniMax-H3 song / speech / instrumental prompts.

H3 generates audio and video jointly in one packed sequence, so audio-only work is
just a normal generation with the canvas pinned tiny. At 32x32 a 60 s render is
~427 video rows against ~4,834 audio rows: the video is a throwaway, the audio is
the product. Wire the sampler output through VAEDecodeAudio and ignore the video
branch entirely.

Nothing in this file touches the model. Generation already works with the stock
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo nodes — what those nodes do NOT
do is build the prompt, and H3's audio prompting has a specific format that leaves
most of the control on the table if you write plain prose:

    integrated_multimodal_description:   the performance. The ONLY field where
                                         <d> blocks are allowed to live.
    overall_soundscape:                  the room it was recorded in.
    non_diegetic_music:                  score the characters CANNOT hear. An
                                         instrument a performer is playing is
                                         DIEGETIC and belongs in the description —
                                         put it here and the model lays an
                                         unrelated backing track underneath.

    <d>[English] ... </d>   ONLY this text is vocalised. Everything else is
                            direction the model uses to decide HOW it sounds.
    [Shot N] At 00:MM.SSS   the trained way to place an event in time. Prose
                            windows ("from 0:16 to 0:29") are off-format and get
                            paced far more loosely.
    (S1)                    speaker tag.

Two things that matter more than they look:

  VOICE DESCRIPTION OUTWEIGHS THE WORDS. "a low breathy smoky whisper with audible
  breath between phrases" and "a bright belting pop voice with a hard edge" sing
  identical lyrics completely differently.

  TIME MUST BE ALLOCATED. Without per-section timing a 60 s prompt gets sung as
  fast as possible and then vamps for forty seconds — the model has no idea the
  bridge is meant to arrive two thirds of the way in. This node splits the
  duration across sections by kind and line count and states it as shot markers.

Pacing rule of thumb: 2-3 s per short sung line. The node warns when a section is
crammed (it will rush or drop the tail) or too loose (it will vamp).

VOICE CLONING is a separate mechanism: MiniMaxH3ReferenceToVideo takes up to three
ref_audios and presents them as <Audio 1>..<Audio 3>. Enable `voice_reference`
here to add phrasing that ties the performer to <Audio 1>. UNVERIFIED as of
2026-08-07 — the risk is that the model replays the sample instead of cloning the
speaker, which is what the retention wording is trying to avoid.
"""

import re

FPS = 24

# how long each section wants, relative to the others
SECTION_WEIGHT = {
    "intro": 1.0, "verse": 2.4, "pre-chorus": 1.2, "chorus": 2.2,
    "bridge": 1.5, "solo": 1.6, "breakdown": 1.4, "outro": 1.2, "hook": 1.8,
}
SECTION_KINDS = ["pre-chorus", "breakdown", "chorus", "verse", "bridge", "intro",
                 "outro", "solo", "hook"]
LEAD_IN = {
    "intro": "an instrumental introduction with no vocal",
    "verse": "the verse",
    "pre-chorus": "the pre-chorus, building",
    "chorus": "the chorus, opening up a little",
    "bridge": "the bridge, stripped back and quieter",
    "solo": "an instrumental solo with no vocal",
    "breakdown": "an instrumental breakdown with no vocal",
    "outro": "the outro, falling away",
    "hook": "the hook",
}
_ORD = ["first", "second", "third", "fourth"]
LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "Spanish", "French",
             "German", "Portuguese", "Italian", "Russian"]


def align_frames(seconds):
    """The video VAE needs frame counts on a 17n+5 grid; out-of-grid snaps down."""
    n = max(5, round(seconds * FPS))
    while n % 17 != 5:
        n += 1
    return n


def stamp(t):
    return f"00:{t:06.3f}" if t < 60 else f"{int(t) // 60:02d}:{t % 60:06.3f}"


def kind_of(section):
    s = (section or "verse").lower()
    for k in SECTION_KINDS:
        if k in s:
            return k
    return "verse"


def parse_sections(text):
    """-> [(section_or_None, [line, ...]), ...]

    A [Header] with no lines under it is kept with an empty list, which is how you
    get a real instrumental intro, solo or outro that plays out.
    """
    out, section, buf, seen = [], None, [], False
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        m = re.fullmatch(r"\[(.+?)\]", line)
        if m:
            if seen:
                out.append((section, buf))
            section, buf, seen = m.group(1).strip().lower(), [], True
            continue
        if line:
            buf.append(line)
    if seen or buf:
        out.append((section, buf))
    return out


def plan_song(text, seconds):
    blocks = parse_sections(text)
    if not blocks:
        return []
    weights = []
    for section, lines in blocks:
        w = SECTION_WEIGHT.get(kind_of(section), 2.0)
        if lines:
            w *= (0.55 + 0.45 * len(lines))     # more lines -> more room
        weights.append(w)
    total = sum(weights) or 1.0
    out, t = [], 0.0
    for (section, lines), w in zip(blocks, weights):
        dur = seconds * w / total
        out.append({"section": section or "verse", "kind": kind_of(section),
                    "lines": lines, "start": t, "end": min(t + dur, seconds)})
        t += dur
    return out


SPEAKER_RE = re.compile(
    r"^(?:(\d+):(\d+(?:\.\d+)?)\s*\|\s*)?(?:\s*(S\d+)\s*[:|]\s*)?(.+)$")


def parse_voices(text):
    """`S1 = description` per line -> {"S1": desc}. Plain prose -> {"S1": prose}.

    One field covers both cases: a monologue just describes a voice, a scene
    assigns one per speaker. Keeps the node from growing three more text boxes.
    """
    text = (text or "").strip()
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(S\d+)\s*=\s*(.+)$", line.strip())
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out or {"S1": text}


def plan_speech(text, seconds):
    """Lines, one per beat. `0:06 | line` pins a line; unpinned lines share what is left.

    Explicit timing is worth using for dialogue — it is the difference between
    lines landing where you want them and the model deciding for you.
    """
    raw = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    items = []
    last_spk = "S1"
    for line in raw:
        m = SPEAKER_RE.match(line)
        if not m:
            continue
        mm, ss, spk, txt = m.groups()
        at = (int(mm) * 60 + float(ss)) if mm is not None else None
        # an untagged line continues whoever spoke last, so a monologue needs no tags
        spk = (spk or last_spk).upper()
        last_spk = spk
        items.append({"at": at, "text": txt.strip(), "spk": spk})
    n = len(items)
    if not n:
        return []
    # spread unpinned lines evenly through the take, leaving a little air at the end
    span = seconds * 0.9
    for i, it in enumerate(items):
        if it["at"] is None:
            it["at"] = span * i / max(1, n)
    items.sort(key=lambda x: x["at"])
    for i, it in enumerate(items):
        it["end"] = items[i + 1]["at"] if i + 1 < n else seconds
    return items


PRESETS = {
    "custom (use fields below)": None,
    "intimate close-mic dialogue": dict(
        mode="speech",
        voice="a low, warm, unhurried voice very close to the microphone, with audible "
              "breath between sentences and a slight smile in the tone",
        room="A small quiet room late at night: faint room tone, one distant car passing "
             "outside, the creak of a chair",
        script="I wasn't going to say anything.\n0:06 | But you already knew that, didn't you.\n"
               "0:12 | It's late. Stay a little longer.",
        style="", instrumentation=""),
    "sultry R&B instrumental": dict(
        mode="instrumental",
        style="A slow, sultry R&B instrumental at about 70 BPM in a minor key, staying at "
              "the same intensity the whole way through",
        instrumentation="A deep round electric bass holding a patient groove, brushed drums "
                        "low and behind the beat, a muted electric guitar answering with short "
                        "clean figures, and a warm electric piano with space between chords",
        room="A dim club room late at night, faint room tone, slight natural reverb, no crowd",
        voice="", script=""),
    "sparse acoustic ballad": dict(
        mode="song",
        style="A sparse, slow, sultry acoustic ballad around 68 BPM, minor key, yearning and "
              "melancholy, with a lot of space between phrases and no big chorus",
        instrumentation="One softly fingerpicked nylon-string guitar and nothing else - no "
                        "drums, no percussion, no bass, no synth and no strings at any point",
        voice="a low, breathy, smoky female voice singing quietly and very close to the "
              "microphone, with audible breath between phrases and lazy phrasing that drags "
              "slightly behind the beat",
        room="A small quiet room late at night: faint room tone, her breath and lip noise "
             "close to the microphone between phrases",
        script="[Intro]\n\n[Verse 1]\nHoney, the hour is late\nHoney, I hate to wait\n\n"
               "[Chorus]\nOh, I want you\nMore than I want to\n\n[Outro]\n"),
    "angry shouted argument": dict(
        mode="speech",
        voice="a raw, raised female voice with a hard rasp, clipped and fast, breathing "
              "hard between sentences",
        room="A bare hallway with hard walls and a short slap-back echo",
        script="Don't you dare walk away from me.\n0:04 | I asked you a question.\n"
               "0:09 | Fine. Go.",
        style="", instrumentation=""),
}


def _line_end(item):
    """Rough finish time of a line — ~2.5 words/second, floor of 1.5 s."""
    return item["at"] + max(1.5, max(1, len(item["text"].split())) / 2.5)


def _speech_end(items):
    return _line_end(items[-1])


def _fit_seconds(mode, script, fallback):
    """Length the script actually needs, so there is no dead air to fill.

    Unclaimed time is where the model invents speech — CJ, 2026-08-07: "she is
    still speaking gibberish in between lines". Sizing the take to the content is
    the cheapest way to remove the problem rather than argue with it in prose.
    """
    if mode == "instrumental":
        return fallback
    if mode == "speech":
        items = plan_speech(script, fallback)
        if not items:
            return fallback
        return round(max(3.0, _speech_end(items) + 1.5), 2)
    blocks = parse_sections(script)
    if not blocks:
        return fallback
    # ~3 s per sung line, ~3 s for an instrumental passage
    total = sum(3.0 * len(lines) if lines else 3.0 for _, lines in blocks)
    return round(max(5.0, total), 2)


class H3AudioPrompt:
    """Build a MiniMax-H3 audio prompt (song / speech / instrumental) + frame count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(PRESETS), {"default": "custom (use fields below)",
                            "tooltip": "Picking a preset fills the fields below so you edit "
                                       "rather than start blank. Set back to custom to keep "
                                       "your own text."}),
                "mode": (["song", "speech", "instrumental"], {"default": "song"}),
                "auto_fit_duration": ("BOOLEAN", {"default": True,
                            "tooltip": "Set the take length from the script itself, so there "
                                       "is no unclaimed time for the model to fill with "
                                       "invented speech. Overrides `seconds`."}),
                "seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0,
                                      "step": 0.5,
                                      "tooltip": "Snaps up to the 17n+5 frame grid. "
                                                 "60 s -> 1450 frames. The ~362 frame "
                                                 "'trained range' is a VIDEO limit; at a "
                                                 "tiny canvas long audio is fine."}),
                "style": ("STRING", {"multiline": True, "default":
                          "A sparse, slow, sultry acoustic ballad around 68 BPM, minor key, "
                          "yearning and melancholy, with a lot of space between phrases"}),
                "instrumentation": ("STRING", {"multiline": True, "default":
                          "One softly fingerpicked nylon-string guitar and nothing else — no "
                          "drums, no percussion, no bass, no synth and no strings at any point",
                          "tooltip": "For a sparse arrangement, name what is NOT there. Left "
                                     "alone the model fills space with drums and pads."}),
                "voice": ("STRING", {"multiline": True, "default":
                          "a low, breathy, smoky female voice very close to the microphone, "
                          "with audible breath between phrases and slow lazy phrasing that "
                          "drags slightly behind the beat",
                          "tooltip": "The single biggest lever on the result — bigger than "
                                     "the words themselves."}),
                "room": ("STRING", {"multiline": True, "default":
                          "A small quiet room late at night: faint room tone, one distant car "
                          "passing outside, breath and lip noise close to the microphone"}),
                "script": ("STRING", {"multiline": True, "default":
                          "[Intro]\n\n[Verse 1]\nHoney, the hour is late\nHoney, I hate to wait\n\n"
                          "[Chorus]\nOh, I want you\nMore than I want to\n\n[Outro]\n",
                          "tooltip": "SONG: [Section] headers, lines beneath. A header with no "
                                     "lines = instrumental passage.\nSPEECH: one line per beat, "
                                     "optionally '0:06 | line text' to pin timing."}),
                "language": (LANGUAGES, {"default": "English"}),
                "speaker": ("STRING", {"default": "S1"}),
                "use_timed_shots": ("BOOLEAN", {"default": True,
                                 "tooltip": "Emit [Shot N] At 00:MM.SSS markers. This is the "
                                            "TRAINED way to place events in time — leaving it "
                                            "off paces loosely. Note [Shot N] means a camera "
                                            "cut, which is free here (32x32 throwaway video) "
                                            "but would cut a real video."}),
            },
            "optional": {
                "voices_from_audio": (["none", "S1", "S1,S2", "S1,S2,S3"],
                                    {"default": "none",
                                    "tooltip": "REPLACES the `voice` field: the speaker's "
                                               "voice comes from ref_audio_1 on the ref2va "
                                               "node instead of your description. Switches the "
                                               "prompt to the six-section reference format. "
                                               "Use CLEAN 5-10s samples — longer references "
                                               "crowd out the target. Speakers map to "
                                               "ref_audio_1/2/3 IN ORDER."}),
                "extra_direction": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "length", "plan")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/audio"
    DESCRIPTION = ("Build a MiniMax-H3 song/speech/instrumental prompt with correct "
                   "field structure, <d> vocal blocks and [Shot N] timing. Feed `prompt` "
                   "to the H3 conditioning node and `length` to its length input; render "
                   "at 32x32 and take VAEDecodeAudio.")

    @staticmethod
    def _mid(text):
        """Lowercase a fragment's first letter — it gets spliced mid-sentence."""
        t = (text or "").strip().rstrip(".")
        return (t[0].lower() + t[1:]) if t and t[0].isupper() and not t[1:2].isupper() else t

    def build(self, preset, mode, auto_fit_duration, seconds, style, instrumentation,
              voice, room, script, language, speaker, use_timed_shots,
              voices_from_audio="none", extra_direction=""):
        shot_markers = use_timed_shots
        cloned = [] if voices_from_audio == "none" else voices_from_audio.split(",")
        voice_reference = bool(cloned)
        pre = PRESETS.get(preset)
        if pre:   # server-side fallback; the JS also writes these into the widgets
            mode = pre["mode"]
            style, instrumentation = pre["style"], pre["instrumentation"]
            voice, room, script = pre["voice"], pre["room"], pre["script"]
        if auto_fit_duration:
            seconds = _fit_seconds(mode, script, seconds)
        instrumentation = self._mid(instrumentation)
        style = self._mid(style)
        frames = align_frames(seconds)
        dur = frames / FPS
        spk = f"({speaker})" if speaker else ""
        subj = "<Subject 1>" if voice_reference else "The speaker"
        # with voice_reference the binding is declared in subject_definitions and
        # retention_analysis; repeating it here just competes with those sections
        who = ("using the voice of <Audio 1>" if voice_reference else self._mid(voice))

        shot = [1]

        def head(t=None):
            if not shot_markers:
                return ""
            if t is None:
                s = f"[Shot {shot[0]}] "
            else:
                s = f" [Shot {shot[0]}] At {stamp(t)},"
            shot[0] += 1
            return s

        lines_out, plan_out = [], []

        if mode == "instrumental":
            body = (f"{head()}A static, dim, close shot that never changes. The audio is "
                    f"{style}, played as one continuous take with no "
                    f"edits and no breaks: {instrumentation}. There is "
                    f"no singing and no speech at any point.")
            plan_out.append(f"0:00-{dur:.2f}  instrumental, no vocal")

        elif mode == "speech":
            items = plan_speech(script, dur)
            voices = parse_voices(voice)
            cast = []
            for _it in items:              # speaker ids, in order of first appearance
                if _it["spk"] not in cast:
                    cast.append(_it["spk"])
            cast = cast or ["S1"]
            self._cast = cast

            def descr(sp):
                # a cloned speaker's voice comes from its <Audio n>, not from prose
                if sp in cloned:
                    return f"using the voice of <Audio {cloned.index(sp) + 1}>"
                return self._mid(voices.get(sp, voices.get("S1", "")))

            def sname(sp):
                i = cast.index(sp)
                if voice_reference:
                    return f"<Subject {i + 1}>"
                return "The speaker" if len(cast) == 1 else f"The {_ORD[i]} speaker"

            if len(cast) == 1:
                body = (f"{head()}A static, dim, close shot that never changes. "
                        f"{sname(cast[0])} speaks alone, {descr(cast[0])}.")
            else:
                who_all = "; ".join(f"{sname(sp)} ({sp}) is {descr(sp)}" for sp in cast)
                body = (f"{head()}A static, dim, close shot that never changes. "
                        f"{len(cast)} people speak, taking turns and never talking over "
                        f"each other: {who_all}.")
            for i, it in enumerate(items):
                d = f"<d>[{language}] {it['text']}</d>"
                # verbatim form from the guide: subject + (Sx) + speech verb + tag.
                # Trailing the id after </d> risks it being vocalised as an artifact.
                sp = it["spk"]
                verb = "says" if (i == 0 or items[i - 1]["spk"] != sp) else "continues"
                if i == 0:
                    body += f" {sname(sp)} ({sp}) {verb}, {d}"
                else:
                    body += f"{head(it['at'])} {sname(sp)} ({sp}) {verb}, {d}"
                plan_out.append(f"{it['at']:6.2f}s  {sp}  {it['text'][:50]}")
                # An unclaimed GAP between two lines gets filled with invented
                # speech, exactly like the unclaimed tail did. Claim it: the pause
                # is an event, so it gets its own shot marker like everything else.
                nxt = items[i + 1]["at"] if i + 1 < len(items) else None
                gap_from = _line_end(it)
                if nxt is not None and nxt - gap_from > 1.8:
                    body += (f"{head(gap_from)} nobody speaks at all: only room tone, no "
                             f"words and no vocal sound of any kind until the next line.")
                    plan_out.append(f"{gap_from:6.2f}s  [pause to {nxt:.2f}s]")
            # An UNTIMED "and then it is quiet" is off-format, and the model fills the
            # unclaimed time with gibberish rather than silence. Give the silence its own
            # shot marker, the same trained mechanism that places the lines.
            tail = _speech_end(items) if items else 0.0
            if items and dur - tail > 1.5:
                body += (f"{head(tail)} no one speaks for the remainder of the take: only "
                         f"the room tone continues to the end, with no voice, no words, no "
                         f"singing, no humming and no music of any kind.")
                plan_out.append(f"{tail:6.2f}s  [silence to {dur:.2f}s]")
            else:
                body += (" They stop speaking and the room is quiet for the rest of the take. "
                         "There is no music and no other voice at any point.")

        else:  # song
            plan = plan_song(script, dur)
            perf = (f"a solo live performance recorded in one take: one performer with "
                    f"{who} sings while playing, accompanied only by "
                    f"{instrumentation}")
            body = (f"{head()}A static, dim, close shot that never changes. The audio is "
                    f"{perf}. {style[0].upper() + style[1:] if style else ''}.")
            first = True
            for p in plan:
                lead = LEAD_IN.get(p["kind"], "a sung section")
                h = "" if first else head(p["start"])
                if not p["lines"]:
                    body += (f"{h} {lead}: they keep playing and do not sing at all during "
                             f"this passage, and the instrument carries it alone."
                             if h else
                             f" The song opens with {lead}: they play alone and do not sing yet.")
                    tag = "instrumental"
                else:
                    sung = " ".join(f"<d>[{language}] {ln}</d>" for ln in p["lines"])
                    verb = "They begin" if first else " they sing"
                    body += (f"{h}{verb} {lead}, spacing these lines evenly and letting "
                             f"the instrument ring between them. {subj} {spk} sings, {sung}")
                    tag = f"{len(p['lines'])} lines"
                first = False
                per = ((p["end"] - p["start"]) / len(p["lines"])) if p["lines"] else 0
                warn = ""
                if p["lines"]:
                    if per < 1.8:
                        warn = f"  ! {per:.1f}s/line — will rush or drop the tail"
                    elif per > 5.0:
                        warn = f"  . {per:.1f}s/line — loose, will vamp"
                plan_out.append(f"{p['start']:6.2f}-{p['end']:6.2f}  "
                                f"{p['section']:<12} {tag}{warn}")
            body += (" They hold the last note and let the final chord ring out and decay to "
                     "silence. There is no other voice and no other instrument at any point.")

        if extra_direction.strip():
            body += " " + extra_direction.strip()

        sound = room.strip() or "N/A"
        if voice_reference:
            # With a reference in play the model needs the SIX-section form from
            # docs/prompting-ref2va.md — the reference must be labelled, given a
            # task type, and given a retention marker. The three-field form works
            # for pure generation but leaves <Audio 1> with no declared role, and
            # the model improvises.
            #
            # `reference` (NOT `fully_copy`) is what asks for timbre and delivery
            # while letting the words be new. fully_copy reuses the signal.
            kind = "sings" if mode == "song" else "speaks"
            cast = getattr(self, "_cast", ["S1"])
            defs, rets = [], []
            for i, sp in enumerate(cast):
                subj_i = f"<Subject {i + 1}>"
                if sp in cloned:
                    a = f"<Audio {cloned.index(sp) + 1}>"
                    defs.append(f"{subj_i} is the {_ORD[i]} speaker ({sp}). "
                                f"{a} is the voice for {subj_i} ({sp}).")
                    rets.append(f"{a}: reference - timbre, accent and delivery only for "
                                f"{subj_i}; the signal is not copied and the words are new.")
                else:
                    defs.append(f"{subj_i} is the {_ORD[i]} speaker ({sp}).")
            names = [f"<Subject {i + 1}>" for i in range(len(cast))]
            who_sum = names[0] if len(names) == 1 else \
                (" and ".join([", ".join(names[:-1]), names[-1]]))
            prompt = (
                f"subject_definitions:\n" + "\n".join(defs) + "\n\n"
                f"summary:\n"
                f"[audio reference] An audio-only take in which {who_sum} "
                f"{kind if len(cast) == 1 else 'speak'} "
                f"new lines, using the referenced audio for voice timbre and delivery "
                f"only. The referenced signals are not reused.\n\n"
                f"retention_analysis:\n" + "\n".join(rets) + "\n\n"
                f"detailed_description:\n{body}\n\n"
                f"overall_soundscape: {sound}\n\n"
                f"non_diegetic_music: N/A")
        else:
            prompt = (f"integrated_multimodal_description: {body}\n\n"
                      f"overall_soundscape: {sound}\n\n"
                      f"non_diegetic_music: N/A")

        header = (f"{mode}  {dur:.2f}s  {frames} frames"
                  f"{'  (voices: ' + voices_from_audio + ')' if voice_reference else ''}")
        plan_text = header + "\n" + "\n".join(plan_out)
        # `ui` goes to the browser so the plan can be drawn on the node body —
        # the pacing warnings are useless if you have to wire an output to read them
        return {"ui": {"h3plan": [plan_text]},
                "result": (prompt, frames, plan_text)}


class H3AudioLength:
    """Seconds -> a legal H3 frame count, for wiring length without the prompt node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seconds": ("FLOAT", {"default": 15.0, "min": 1.0,
                                                   "max": 120.0, "step": 0.5})}}
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("length", "actual_seconds")
    FUNCTION = "go"
    CATEGORY = "MiniMax H3/audio"
    DESCRIPTION = "Snap a duration up to the video VAE's 17n+5 frame grid."

    def go(self, seconds):
        n = align_frames(seconds)
        return (n, n / FPS)


NODE_CLASS_MAPPINGS = {
    "H3AudioPrompt": H3AudioPrompt,
    "H3AudioLength": H3AudioLength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioPrompt": "MiniMax H3 Audio Prompt (song/speech)",
    "H3AudioLength": "MiniMax H3 Audio Length",
}
