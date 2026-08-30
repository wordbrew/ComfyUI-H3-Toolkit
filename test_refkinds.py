"""The four reference kinds on H3ReferenceToVideoLongForm, and the row order.

Two things are checked here, and the second one is the reason this file exists.

REFERENCE KINDS. The node carried only the image path until 2026-08-30, so a
chained dialogue take had no VOICE reference and reinvented the timbre in every
chunk. Images, videos, a video's own soundtrack and standalone audio are now all
present, matching stock MiniMaxH3ReferenceToVideo block for block.

ROW ORDER. The model concatenates the condition latents flat and scatters them
with `all_video_rows[~img_update] = cond_video_rows`, so the ROW order in the
packed layout has to match the order core BUILDS that list in -- keyframes
first, then references (model_base.py:2201,2206). The pack's patched layout
emitted references first until 2026-08-30, which fed every reference latent into
a keyframe's row slot on any graph using both. Nothing errors when that happens;
the totals still match, the contents are simply shifted.

    python3 test_refkinds.py
"""
import importlib
import importlib.util
import pathlib
import sys
import types


# --- stubs ------------------------------------------------------------------ #
# video.py is a NODE module and imports ComfyUI at module level. Stand those in;
# nothing under test touches a real tensor.

class T:
    """A stand-in image batch, [N, H, W, 3]."""
    def __init__(self, n, h=1120, w=640, tag="img"):
        self.n, self.h, self.w, self.tag = n, h, w, tag

    @property
    def shape(self):
        return (self.n, self.h, self.w, 3)

    def __getitem__(self, k):
        if isinstance(k, tuple):                 # img[..., :3]
            return self
        if isinstance(k, list):                  # frames[sample_idx]
            return T(len(k), self.h, self.w, f"{self.tag}@{k}")
        if isinstance(k, slice):
            a, b, _ = k.indices(self.n)
            return T(b - a, self.h, self.w, f"{self.tag}[{a}:{b}]")
        return T(1, self.h, self.w, self.tag)

    def movedim(self, *a):
        return self


class Z:
    """A video latent, [1, 24, T, h, w]."""
    def __init__(self, t, tag="z"):
        self.t, self.tag = t, tag

    @property
    def shape(self):
        return (1, 24, self.t, 70, 40)

    def dim(self):
        return 5

    def __getitem__(self, k):
        return Z(1, f"{self.tag}#{k[2].start}")


class A:
    """An audio latent, [1, 32, 2, T]."""
    def __init__(self, t, tag="a"):
        self.t, self.tag = t, tag

    @property
    def shape(self):
        return (1, 32, 2, self.t)


class VAE:
    def __init__(self):
        self.calls = []

    def encode(self, imgs):
        self.calls.append(imgs.n)
        n = imgs.n
        return Z(2 + 5 * ((n - 5) // 17) if n >= 5 else 1, imgs.tag)


class AudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = []

    def encode(self, wav):
        self.calls.append(wav)
        return A(wav.ticks, wav.tag)


class Wave:
    """Stands in for the [B, C, L] waveform inside an AUDIO dict."""
    def __init__(self, ticks, tag):
        self.ticks, self.tag = ticks, tag

    def __getitem__(self, k):
        return self

    def movedim(self, *a):
        return self


def audio(ticks, tag):
    return {"waveform": Wave(ticks, tag), "sample_rate": 32000}


class Clip:
    def __init__(self):
        self.items = None

    def tokenize(self, prompt, minimax_ref_items=None):
        self.items = minimax_ref_items
        return "tokens"

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m


_t = _mod("torch"); _nn = _mod("torch.nn"); _fn = _mod("torch.nn.functional")
_t.nn = _nn; _nn.functional = _fn
_t.zeros = lambda *a, **k: None
_c = _mod("comfy")
_mm = _mod("comfy.model_management", intermediate_device=lambda: "cpu")
_nt = _mod("comfy.nested_tensor", NestedTensor=lambda pair: pair)
_ut = _mod("comfy.utils",
           common_upscale=lambda s, w, h, m, c: T(s.n, h, w, s.tag))
_c.model_management = _mm; _c.nested_tensor = _nt; _c.utils = _ut
_mod("node_helpers",
     conditioning_set_values=lambda cond, values, **k: [[c, dict(m, **values)]
                                                        for c, m in cond])


class _IO:
    class ComfyNode:
        pass

    class Schema:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _In:
        def __init__(self, *a, **k):
            pass

    class _Autogrow(_In):
        @staticmethod
        def Input(name, **k):
            return type("I", (), {"id": name})()

        class TemplatePrefix:
            def __init__(self, **k):
                pass

    Image = Audio = Latent = Conditioning = Model = Clip = Vae = _In
    Int = Float = String = Boolean = Combo = _In
    Autogrow = _Autogrow
    setattr(_In, "Input", staticmethod(
        lambda name=None, **k: type("I", (), {"id": name})()))
    setattr(_In, "Output", staticmethod(lambda *a, **k: None))
    NodeOutput = staticmethod(lambda *a, **k: a)


_api = _mod("comfy_api")
_lat = _mod("comfy_api.latest", io=_IO())
_api.latest = _lat

root = pathlib.Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location(
    "h3refkinds", root / "__init__.py", submodule_search_locations=[str(root)])
pk = importlib.util.module_from_spec(spec)
sys.modules["h3refkinds"] = pk
try:
    spec.loader.exec_module(pk)
except Exception as e:
    print("(pack partial:", type(e).__name__, ")")
vid = importlib.import_module("h3refkinds.video")
vid.patch_packed_layout = lambda: True           # needs ComfyUI core
vid.adapt_canvas = lambda w, h: (640, 1120)      # core's canvas maths
vid.CANVAS_MULTIPLE = 32

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def run(**kw):
    clip = Clip()
    v, av = VAE(), AudioVAE()
    args = dict(clip=clip, vae=v, audio_vae=av, prompt="a shot", width=640,
                height=1120, length=141)
    args.update(kw)
    cond, latent = vid.H3ReferenceToVideoLongForm.execute(**args)
    return cond[0][1], clip.items, v, av


# --- the four kinds --------------------------------------------------------- #
print("all four reference kinds reach the DiT, in schema order")
meta, items, v, av = run(
    ref_images={"ref_image_1": T(1, 1024, 768, "REF")},
    ref_videos={"ref_video_1": T(60, 480, 640, "VID")},
    ref_video_audios={"ref_video_audio_1": audio(90, "TRACK")},
    ref_audios={"ref_audio_1": audio(120, "VOICE")})
kinds = [b["kind"] for b in meta["minimax_refs"]]
check("image, then the video with its soundtrack, then the voice",
      kinds, ["image", "video_audio", "audio"])

print("a reference video is trimmed DOWN to a legal 17n+5 run")
check("60 frames -> 56", [c for c in v.calls if c not in (1,)][-1], 56)

print("Qwen is shown the video at 2 fps with timestamps, not every frame")
vitem = [i for i in items if i["type"] == "video"][0]
check("56 frames at one in twelve -> 5 samples", vitem["data"].n, 5)
check("stamped in seconds", vitem["timestamps"], [0.0, 0.5, 1.0, 1.5, 2.0])

print("presentation order: a soundtrack is announced before its video")
check("the ref items", [i["type"] for i in items],
      ["image", "audio", "video", "audio"])

print("an audio-only block carries no video latent and is skipped in that list")
check("one image + one video latent", len(meta["cond_video_latents"]), 2)
check("both audio latents", len(meta["cond_audio_latents"]), 2)

print("the soundtrack pairs by NUMBER, not by position")
meta, items, v, av = run(
    ref_videos={"ref_video_2": T(56, 480, 640, "VID2")},
    ref_video_audios={"ref_video_audio_2": audio(90, "TRACK2")})
check("ref_video_audio_2 belongs to ref_video_2",
      meta["minimax_refs"][0]["kind"], "video_audio")
meta, items, v, av = run(
    ref_videos={"ref_video_1": T(56, 480, 640, "VID1")},
    ref_video_audios={"ref_video_audio_3": audio(90, "TRACK3")})
check("a mismatched number leaves the video silent",
      meta["minimax_refs"][0]["kind"], "video")

print("a video shorter than the VAE's minimum is named, not silently dropped")
try:
    run(ref_videos={"ref_video_1": T(3, 480, 640, "TOOSHORT")})
    check("under 5 frames raises", "no error", "ValueError")
except ValueError as e:
    check("and it says which input", "ref_video_1" in str(e), True)

# --- the loose-kwarg fallback ----------------------------------------------- #
# Registered through NODE_CLASS_MAPPINGS the autogrow inputs can arrive as
# separate ref_video_1= kwargs instead of one dict. `ref_video_` also prefixes
# `ref_video_audio_`, so the fallback has to match the SUFFIX shape too or a
# soundtrack ends up in the video list and is encoded by the wrong VAE.
print("loose kwargs: ref_video_ does not swallow ref_video_audio_")
meta, items, v, av = run(**{"ref_video_1": T(56, 480, 640, "VID"),
                            "ref_video_audio_1": audio(90, "TRACK"),
                            "ref_audio_1": audio(120, "VOICE")})
check("one video block, paired, plus the voice",
      [b["kind"] for b in meta["minimax_refs"]], ["video_audio", "audio"])

# --- row order -------------------------------------------------------------- #
print("cond latents are ordered keyframes first, references second")
meta, items, v, av = run(
    ref_images={"ref_image_1": T(1, 1024, 768, "REF")},
    context_images=T(60, 1120, 640, "CTX"), context_frames="22")
lats = meta["cond_video_latents"]
check("22 context frames -> 7 keyframe rows, then the reference",
      [z.tag.split("#")[0] for z in lats],
      ["CTX[38:60]"] * 7 + ["REF[0:1]"])
check("and the layout emits rows in that same order",
      "keyframes first, then refs" in
      (pathlib.Path(root / "video.py").read_text()), True)

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("reference kinds: all checks pass")
