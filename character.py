"""H3 Character library — save a cast member once, recall it by name.

Reference prompting has a lot of repeated setup: the same anchor images, the same
voice sample, and the same wording describing who the person is. Getting any of it
subtly different between renders is how identity drifts for reasons that have
nothing to do with the model. A character bundles all of it under one name.

Layout mirrors the engine's companion registry (h3_engine/companions.py) so a
character is portable between the PC and the fleet:

    models/h3_characters/<name>/
        images/001.png ...        anchor images, in order -> <Picture 1..N>
        voice.wav                 optional voice sample -> <Audio 1>
        card.json                 description, voice description, retention marker

WHAT THE CACHE HERE DOES AND DOES NOT DO
  This caches decoded image tensors and the audio waveform, keyed on file identity,
  so repeated queue runs skip disk and decode. That is all it can do from a node.

  The expensive parts live inside the conditioning node: the VAE encode of each
  anchor, the vision tower, and the text-encoder forward. Our engine caches the
  first two (h3_engine/conditioning_cache.py) but NOT the third — H3 splices
  references and prompt into one document, so any prompt change reruns the language
  model regardless. And nothing caches the per-step cost of reference tokens riding
  through sampling, which is the larger number: `ref_image_size="max"` measured
  1.78x slower than `match`, almost all of it per-step rather than encode.

  So the real value here is CONSISTENCY, not speed. Treat any speedup as a bonus.

ANCHOR COUNT: 3 anchors beat 1 (measured, Q1). ref2va takes up to 9 images and 3
audio references.
"""

import hashlib
import json
import os
import subprocess

import numpy as np
import torch

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_SLOTS = 5           # separate IMAGE outputs; ref2va accepts up to 9
_CACHE = {}             # (path, mtime, size) -> tensor


def characters_dir(create=False):
    """models/h3_characters. Only CREATES it when `create` is set.

    It used to mkdir unconditionally, and `list_characters()` calls this to fill
    the character dropdown -- which `INPUT_TYPES` calls, which ComfyUI calls on
    every /object_info request. So opening the graph wrote to disk. Saving is the
    only operation that needs the directory to exist.
    """
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    d = os.path.join(base, "h3_characters")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def list_characters():
    d = characters_dir()
    if not os.path.isdir(d):          # nothing saved yet, and nothing to create
        return ["(no characters saved)"]
    out = [n for n in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, n))]
    return out or ["(no characters saved)"]


def _key(path):
    st = os.stat(path)
    return (path, st.st_mtime_ns, st.st_size)


def _load_image(path):
    """-> [1, H, W, 3] float32 in [0,1], cached on file identity."""
    k = _key(path)
    if k in _CACHE:
        return _CACHE[k]
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    _CACHE[k] = t
    return t


def _load_audio(path):
    """-> ComfyUI AUDIO dict. ffmpeg fallback: torchaudio needs torchcodec in some
    builds and raises on load, which is not worth failing a whole graph over."""
    k = _key(path)
    if k in _CACHE:
        return _CACHE[k]
    wav = sr = None
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
    except Exception:
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "2",
             "-ar", "32000", "-"], capture_output=True).stdout
        if raw:
            a = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
            wav, sr = torch.from_numpy(a), 32000
    if wav is None:
        return None
    if wav.shape[0] == 1:            # the audio VAE is stereo; mono encodes malformed
        wav = wav.repeat(2, 1)
    out = {"waveform": wav[:2].unsqueeze(0), "sample_rate": int(sr)}
    _CACHE[k] = out
    return out


def read_card(name):
    p = os.path.join(characters_dir(), name, "card.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def sanitise(name):
    """The store's folder name for a subject. Raises on a name that empties."""
    safe = "".join(c for c in str(name).strip()
                   if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise ValueError("subject name is empty after sanitising")
    return safe


def card_of(name, kind="person", description="", voice_description="",
            retention="fully_preserved", anchors=0):
    """The card dict, in ONE place, so the node and the HTTP route agree."""
    return {"name": name, "kind": kind, "description": str(description).strip(),
            "voice": str(voice_description).strip(), "retention": retention,
            "anchors": int(anchors)}


def save_subject(name, kind="person", description="", voice_description="",
                 retention="fully_preserved", image_paths=(), voice_path=None,
                 overwrite=False):
    """Write a subject to the store from FILES on disk. -> the card.

    The node's save() takes IMAGE tensors because that is what a graph carries.
    An HTTP caller has filenames -- ComfyUI's own /upload/image already put them
    in input/ -- so copying beats a tensor round trip. Both land the identical
    layout: card.json, images/NNN.png, voice.wav. `card_of` is shared so the two
    cannot write different cards.
    """
    import shutil
    safe = sanitise(name)
    d = os.path.join(characters_dir(create=True), safe)
    if os.path.isdir(d) and not overwrite:
        raise ValueError(f"'{safe}' already exists — pass overwrite to replace it")
    img_dir = os.path.join(d, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    n = 0
    for src in image_paths or ():
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(src)[1].lower()
        if ext not in IMAGE_EXT:
            continue
        n += 1
        shutil.copyfile(src, os.path.join(img_dir, f"{n:03d}{ext}"))
    if voice_path and os.path.isfile(voice_path):
        shutil.copyfile(voice_path,
                        os.path.join(d, "voice" + os.path.splitext(voice_path)[1].lower()))
    card = card_of(safe, kind, description, voice_description, retention, n)
    with open(os.path.join(d, "card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    _CACHE.pop(safe, None)
    return card


def stage_subject(name, input_dir, folder="h3_subjects"):
    """Copy a subject's anchors into ComfyUI's INPUT folder. -> [relative paths]

    WHY THIS EXISTS
      A subject's anchors live in models/h3_characters/<name>/images/, and
      LoadImage only lists ComfyUI/input. So a cast member picked from the
      library had its pictures CITED in the prompt while the images actually
      encoded came from somewhere else entirely -- the "five nodes that have to
      agree" failure, with the two halves in different directories.

      Copying into input/ makes the anchors reachable by the same node everything
      else uses, so the pictures cited and the pictures sent are the same files.
      It also puts them in the asset gallery, where they can be seen.

    IT IS A CACHE, NOT A SECOND COPY OF THE TRUTH
      The store stays authoritative. This is idempotent -- same name, same
      destination, overwritten each time -- so re-staging after an edit refreshes
      it, and nothing downstream has to know whether it ran.
    """
    import shutil
    safe = sanitise(name)
    src = os.path.join(characters_dir(), safe, "images")
    if not os.path.isdir(src):
        return []
    dest = os.path.join(str(input_dir), folder, safe)
    os.makedirs(dest, exist_ok=True)
    out = []
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(IMAGE_EXT):
            continue
        shutil.copyfile(os.path.join(src, f), os.path.join(dest, f))
        # LoadImage lists nested files with forward slashes
        out.append(f"{folder}/{safe}/{f}")
    return out


def delete_subject(name):
    """Remove a subject from the store. -> True if it was there."""
    import shutil
    safe = sanitise(name)
    d = os.path.join(characters_dir(), safe)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d)
    _CACHE.pop(safe, None)
    return True


def list_subjects():
    """Every saved subject with its card, for a UI that must SHOW the library."""
    d = characters_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not os.path.isdir(os.path.join(d, name)):
            continue
        card = read_card(name) or {}
        img_dir = os.path.join(d, name, "images")
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.lower().endswith(IMAGE_EXT)) if os.path.isdir(img_dir) else []
        out.append({"name": name,
                    "kind": str(card.get("kind") or "person").lower(),
                    "description": card.get("description", ""),
                    "voice": card.get("voice", ""),
                    "retention": card.get("retention", "fully_preserved"),
                    "anchors": int(card.get("anchors", len(imgs)) or len(imgs)),
                    "images": imgs})
    return out


class H3Character:
    """Load a saved character: anchors, voice, and the wording that describes them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": (list_characters(), {"tooltip":
                               "A character saved by H3 Character Save, read from "
                               "models/h3_characters. The list is built when "
                               "ComfyUI asks for node definitions, so a character "
                               "saved in this session needs a browser refresh to "
                               "appear. Everything else on this node describes how "
                               "to CITE it; the appearance and voice wording come "
                               "from its card."}),
                "anchors": ("INT", {"default": 3, "min": 1, "max": MAX_SLOTS,
                            "tooltip": "How many anchor images to output. 3 measurably "
                                       "beat 1 for identity; more costs per-step time "
                                       "because reference tokens ride every step."}),
                "subject_index": ("INT", {"default": 1, "min": 1, "max": 9,
                                  "tooltip": "Which <Subject N> this character is, for the "
                                             "generated wording. Set 2 for a second cast "
                                             "member so the tags do not collide."}),
                "picture_offset": ("INT", {"default": 0, "min": 0, "max": 8,
                                   "tooltip": "First <Picture N> number. With two characters, "
                                              "give the second an offset equal to the first "
                                              "one's anchor count."}),
                "audio_index": ("INT", {"default": 1, "min": 1, "max": 3,
                                "tooltip": "Which <Audio N> the voice is, matching which "
                                           "ref_audio slot you wire it into."}),
            }
        }

    RETURN_TYPES = tuple(["IMAGE"] * MAX_SLOTS) + ("AUDIO", "STRING", "STRING",
                                                   "STRING", "STRING")
    RETURN_NAMES = tuple(f"image_{i + 1}" for i in range(MAX_SLOTS)) + (
        "voice", "subject_def", "retention", "description", "info")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/character"
    DESCRIPTION = ("Recall a saved character: anchor images, voice sample, and ready-made "
                   "subject_definitions / retention_analysis wording. Outputs beyond the "
                   "character's anchor count are None — wire only as many as `info` reports.")

    def load(self, character, anchors, subject_index, picture_offset, audio_index):
        d = os.path.join(characters_dir(), character)
        card = read_card(character)
        imgs = []
        img_dir = os.path.join(d, "images")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(IMAGE_EXT):
                    imgs.append(_load_image(os.path.join(img_dir, f)))
        imgs = imgs[:anchors]

        voice = None
        for cand in ("voice.wav", "voice.flac", "voice.mp3"):
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                voice = _load_audio(p)
                break

        pics = [f"<Picture {picture_offset + i + 1}>" for i in range(len(imgs))]
        pic_list = (" and ".join([", ".join(pics[:-1]), pics[-1]]) if len(pics) > 1
                    else (pics[0] if pics else ""))
        subj = f"<Subject {subject_index}>"
        desc = (card.get("description") or f"the person in {pic_list}").strip()

        # FOLD the pictures into the subject definition; never as their own
        # sentence. docs/prompting-ref2va.md: "<Picture N> standalone ONLY when
        # the image anchors a specific shot frame (first/last/keyframe);
        # otherwise fold it into a <Subject N> definition." The template stub is
        # one sentence: "<Subject 1> is [description; from <Picture 1>]."
        #
        # Emitting "<Subject 1>'s appearance is given by <Picture 1>." as a
        # standalone sentence therefore told the model those anchors WERE frame
        # anchors, and it obliged -- a reference showing up as the opening frame
        # (WF 22, 2026-09-02).
        first = f"{subj} is {desc.rstrip('.')}"
        if pics:
            first += f", from {pic_list}"
        lines = [first + "."]
        if voice is not None:
            lines.append(f"<Audio {audio_index}> is the voice for {subj} (S{subject_index}).")
        subject_def = "\n".join(lines)

        marker = card.get("retention", "fully_preserved")
        # Bind retention to the SUBJECT, citing the pictures only as the source of
        # their attributes, and say which shot the subject appears in. Naming
        # <Picture N> as the retained thing makes the model render the anchor image
        # AS A SHOT once motion context is present — a hard cut mid-clip to the
        # anchor's own studio background (090 cut at 7.71s/9.79s; 091 with this
        # wording and nothing else changed was clean at two seeds, one of them the
        # seed that had cut). Matches the working 14-clip one-take workflow.
        # WHAT "KEEP IT THE SAME" MEANS DEPENDS ON WHAT IT IS. This used to be
        # hardcoded person wording, so a saved PLACE came back asking the model
        # to preserve its "facial identity, hairstyle, body proportions" --
        # H3CharacterSave has written `kind` since it gained the widget, and the
        # loader simply never read it. h3script fixed the same bug in its own
        # path months ago; this shares that table rather than restating it, so
        # the two cannot drift again.
        from .h3script import PRESERVE_BY_KIND
        kind = str(card.get("kind") or "person").lower()
        preserve = PRESERVE_BY_KIND.get(kind, PRESERVE_BY_KIND["person"])
        allow = ("natural poses and expressions" if kind == "person" else
                 "natural changes of light and viewpoint" if kind in ("place", "setting")
                 else "natural handling and viewpoint")
        rets = []
        if pics:
            rets.append(f"{subj} (appears in [Shot 1]): {marker} - preserve "
                        f"{preserve} from {pic_list} while allowing {allow}.")
        if voice is not None:
            rets.append(f"<Audio {audio_index}>: reference - timbre, accent and delivery "
                        f"only for {subj}; the signal is not copied and the words are new.")
        retention = "\n".join(rets)

        info = (f"{character}: {len(imgs)} anchor(s) as {pic_list or 'none'}"
                f"{', voice' if voice is not None else ', NO voice'}"
                f" | wire image_1..image_{len(imgs)}")
        if len(imgs) < anchors:
            info += f" | asked for {anchors}, only {len(imgs)} on disk"

        slots = [imgs[i] if i < len(imgs) else None for i in range(MAX_SLOTS)]
        return {"ui": {"h3char": [info]},
                "result": tuple(slots) + (voice, subject_def, retention, desc, info)}


class H3CharacterSave:
    """Write a character to the library so it can be recalled by name."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "new_character",
                         "tooltip": "The library name, and the folder it is "
                                    "written to. Sanitised to letters, digits, "
                                    "spaces, - and _ , so a name that reduces to "
                                    "nothing is rejected. Saving over an existing "
                                    "name needs `overwrite`."}),
                "description": ("STRING", {"multiline": True, "default":
                                "a woman with curly copper-red hair, freckled fair skin "
                                "and a curvy figure",
                                "tooltip": "Goes into subject_definitions as '<Subject N> "
                                           "is <this>.' Describe appearance, not plot."}),
                "voice_description": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Used when you are NOT cloning from a "
                                                 "sample. Voice wording outweighs the "
                                                 "words themselves."}),
                # WHAT IS THIS THING? The store held people and nothing else,
                # so a saved lamp came back with "preserve facial identity, hair,
                # eye colour and build". The wording that keeps a reference
                # consistent is not the same wording for a person, a place and an
                # object, and only the thing itself knows which it is.
                "kind": (["person", "place", "thing"], {"default": "person",
                         "tooltip": "Decides how the prompt asks for it to be "
                                    "preserved. A person keeps identity and "
                                    "build; a place keeps layout and light; a "
                                    "thing keeps shape, colour and markings."}),
                "retention": (["fully_preserved", "partially_preserved",
                               "attribute_transfer", "weak_reference"],
                              {"default": "fully_preserved",
                               "tooltip": "The retention_analysis MARKER stored "
                                          "on the card, which H3 Character writes "
                                          "into the prompt as "
                                          "'<Subject N>: <marker> - preserve their "
                                          "facial identity, hairstyle...'.\n\n"
                                          "fully_preserved is the one the working "
                                          "14-clip take used. The weaker three "
                                          "ask the model to treat the anchors as "
                                          "inspiration rather than identity, which "
                                          "is what you want for a style or a "
                                          "body-type reference and NOT for a "
                                          "character who has to stay the same "
                                          "person across links."}),
                "overwrite": ("BOOLEAN", {"default": False,
                               "tooltip": "Replace an existing character of this "
                                          "name. OFF raises rather than "
                                          "overwriting, so a re-queue cannot "
                                          "quietly destroy a card. ON deletes the "
                                          "stored images before writing the new "
                                          "ones, so saving with fewer anchors "
                                          "really does leave fewer."}),
            },
            "optional": {
                **{f"image_{i + 1}": ("IMAGE", {"tooltip":
                     (f"Anchor image {i + 1} of {MAX_SLOTS}, written into the "
                      f"character's images/ folder. Three anchors measurably beat "
                      f"one for identity, and reference tokens ride EVERY step, so "
                      f"more is not free. Vary angle and lighting; near-duplicate "
                      f"frames spend tokens without adding information."
                      if i == 0 else
                      f"Anchor image {i + 1} of {MAX_SLOTS}. Slots are collected in "
                      f"order and gaps are closed, so wiring 1, 2 and 4 saves three "
                      f"anchors.")})
                   for i in range(MAX_SLOTS)},
                "voice": ("AUDIO", {"tooltip": "A clean voice SAMPLE to clone "
                           "from, stored with the card and cited as <Audio N>. "
                           "Ref2VA takes 2-15s per audio reference and an audio "
                           "reference must be paired with an image or video.\n\n"
                           "With a sample the prompt says timbre, accent and "
                           "delivery are referenced and the words are new — the "
                           "signal is not copied. Without one, "
                           "`voice_description` carries the voice instead, and "
                           "wording there outweighs the words being said."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/character"
    DESCRIPTION = ("Save anchors + voice + wording under a name, into "
                   "models/h3_characters/. Refresh the browser afterwards for it to "
                   "appear in the H3 Character dropdown.")

    def save(self, name, description, voice_description, retention, overwrite,
             kind="person", voice=None, **images):
        safe = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ").strip()
        if not safe:
            raise ValueError("character name is empty after sanitising")
        d = os.path.join(characters_dir(create=True), safe)
        if os.path.isdir(d) and not overwrite:
            raise ValueError(f"'{safe}' already exists — tick overwrite to replace it")
        img_dir = os.path.join(d, "images")
        os.makedirs(img_dir, exist_ok=True)
        for f in os.listdir(img_dir):
            os.remove(os.path.join(img_dir, f))

        from PIL import Image
        n = 0
        for i in range(MAX_SLOTS):
            t = images.get(f"image_{i + 1}")
            if t is None:
                continue
            for b in range(t.shape[0]):          # an IMAGE input may be a batch
                arr = (t[b].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                n += 1
                Image.fromarray(arr).save(os.path.join(img_dir, f"{n:03d}.png"))

        have_voice = False
        if voice is not None:
            w = voice["waveform"]
            w = w[0] if w.dim() == 3 else w
            if w.shape[0] == 1:
                w = w.repeat(2, 1)
            import wave
            a = (np.clip(w.cpu().numpy(), -1, 1) * 32767).astype(np.int16).T.tobytes()
            with wave.open(os.path.join(d, "voice.wav"), "wb") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(int(voice.get("sample_rate", 32000)))
                f.writeframes(a)
            have_voice = True
            secs = w.shape[-1] / max(1, int(voice.get("sample_rate", 32000)))
            if secs > 12:
                print(f"[H3Character] '{safe}' voice is {secs:.0f}s — 5-10s is plenty. "
                      f"A long reference competes with the target for audio tokens.")

        with open(os.path.join(d, "card.json"), "w", encoding="utf-8") as f:
            json.dump({"name": safe, "kind": kind,
                       "description": description.strip(),
                       "voice": voice_description.strip(), "retention": retention,
                       "anchors": n}, f, indent=2)

        info = f"saved '{safe}': {n} anchor(s), {'voice' if have_voice else 'no voice'} -> {d}"
        print("[H3Character] " + info)
        return {"ui": {"h3char": [info]}, "result": (info,)}


NODE_CLASS_MAPPINGS = {"H3Character": H3Character, "H3CharacterSave": H3CharacterSave}
# THE IDS DO NOT CHANGE. `H3Character` is what every saved workflow references
# and `models/h3_characters/` is where the assets live -- renaming either would
# orphan both. Only the labels move, because the store holds places and props now
# and "Character" had stopped being true.
NODE_DISPLAY_NAME_MAPPINGS = {"H3Character": "H3 Reference",
                              "H3CharacterSave": "H3 Reference (save)"}
