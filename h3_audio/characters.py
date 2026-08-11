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


def characters_dir():
    """models/h3_characters, created on demand."""
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    d = os.path.join(base, "h3_characters")
    os.makedirs(d, exist_ok=True)
    return d


def list_characters():
    d = characters_dir()
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


class H3Character:
    """Load a saved character: anchors, voice, and the wording that describes them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": (list_characters(),),
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

        lines = [f"{subj} is {desc}."]
        if pics:
            lines.append(f"{subj}'s appearance is given by {pic_list}.")
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
        rets = []
        if pics:
            rets.append(f"{subj} (appears in [Shot 1]): {marker} - preserve their facial "
                        f"identity, hairstyle, body proportions, skin and distinctive "
                        f"features from {pic_list} while allowing natural poses and "
                        f"expressions.")
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
                "name": ("STRING", {"default": "new_character"}),
                "description": ("STRING", {"multiline": True, "default":
                                "a woman with curly copper-red hair, freckled fair skin "
                                "and a curvy figure",
                                "tooltip": "Goes into subject_definitions as '<Subject N> "
                                           "is <this>.' Describe appearance, not plot."}),
                "voice_description": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Used when you are NOT cloning from a "
                                                 "sample. Voice wording outweighs the "
                                                 "words themselves."}),
                "retention": (["fully_preserved", "partially_preserved",
                               "attribute_transfer", "weak_reference"],
                              {"default": "fully_preserved"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                **{f"image_{i + 1}": ("IMAGE",) for i in range(MAX_SLOTS)},
                "voice": ("AUDIO",),
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
             voice=None, **images):
        safe = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ").strip()
        if not safe:
            raise ValueError("character name is empty after sanitising")
        d = os.path.join(characters_dir(), safe)
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
            json.dump({"name": safe, "description": description.strip(),
                       "voice": voice_description.strip(), "retention": retention,
                       "anchors": n}, f, indent=2)

        info = f"saved '{safe}': {n} anchor(s), {'voice' if have_voice else 'no voice'} -> {d}"
        print("[H3Character] " + info)
        return {"ui": {"h3char": [info]}, "result": (info,)}


NODE_CLASS_MAPPINGS = {"H3Character": H3Character, "H3CharacterSave": H3CharacterSave}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Character": "H3 Character",
                              "H3CharacterSave": "H3 Character (save)"}
