"""H3MotionContext — the step offsets, the grid snap, and coexistence.

The three things that were wrong for nine waves engine-side, per the wave log:
an off-grid context length that snapped DOWN and covered the wrong frames; a
per-frame encode with no temporal structure; and slicing a previous run's latent
so the steps carried the wrong positional structure. Only the first two are
testable without torch; the third is a design property (this node takes PIXELS).
"""
import sys, types, importlib.util, pathlib

class Z:
    """A latent standing in for one VAE encode: [1, 24, T, h, w]."""
    def __init__(self, t, tag="z"): self.t, self.tag = t, tag
    @property
    def shape(self): return (1, 24, self.t, 70, 40)
    def dim(self): return 5
    def __getitem__(self, k): return Z(1, f"{self.tag}[step {k[2].start}]")

class Img:
    def __init__(self, n, tag="ctx"): self.n, self.tag = n, tag
    @property
    def shape(self): return (self.n, 1120, 640, 3)
    def __getitem__(self, k):
        a, b, _ = k.indices(self.n) if isinstance(k, slice) else (0, self.n, 1)
        return Img(b - a, f"{self.tag}[{a}:{b}]")

class VAE:
    def __init__(self): self.calls = []
    def encode(self, imgs):
        self.calls.append(imgs.n)
        n = imgs.n
        return Z(2 + 5 * ((n - 5) // 17) if n >= 5 else 1)

# video.py is a NODE module, so it imports ComfyUI at module level. Stand those
# in — the arithmetic under test touches none of them.
def _mod(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m
_t = _mod("torch"); _nn = _mod("torch.nn"); _fn = _mod("torch.nn.functional")
_t.nn = _nn; _nn.functional = _fn
_c = _mod("comfy"); _mm = _mod("comfy.model_management"); _c.model_management = _mm
_mod("node_helpers", conditioning_set_values=lambda c, v, **k: c)
class _IO:
    class ComfyNode: pass
    class Schema:
        def __init__(self, **kw): self.__dict__.update(kw)
    class _In:
        def __init__(self, *a, **k): pass
    Image = Audio = Latent = Conditioning = Model = Clip = Vae = _In
    Int = Float = String = Boolean = Combo = AutogrowDynamic = _In
    for _n in ("Input", "Output"):
        setattr(_In, _n, staticmethod(lambda *a, **k: None))
    NodeOutput = staticmethod(lambda *a, **k: a)
_api = _mod("comfy_api"); _lat = _mod("comfy_api.latest", io=_IO()); _api.latest = _lat
root = pathlib.Path("/home/cgree/projects/ComfyUI-H3-Toolkit")
spec = importlib.util.spec_from_file_location("h3v", root/"__init__.py",
                                              submodule_search_locations=[str(root)])
pk = importlib.util.module_from_spec(spec); sys.modules["h3v"] = pk
try: spec.loader.exec_module(pk)
except Exception as e: print("(pack partial:", type(e).__name__, ")")
import importlib
vid = importlib.import_module("h3v.video")
vid.patch_packed_layout = lambda: True          # needs ComfyUI core

fails = []
def check(l, got, want):
    if got != want: fails.append(f"{l}: got {got!r}, want {want!r}"); print("  FAIL", l, got, want)
    else: print("  ok  ", l)

def R(x):
    """The node returns a bare tuple on passthrough and a ui dict otherwise."""
    return x["result"] if isinstance(x, dict) else x


mc = vid.H3MotionContext()
COND = [["CROSSATTN", {"minimax_refs": [{"latent": "R1"}, {"latent": "R2"}],
                       "minimax_keyframes": [{"resolved_frame_index": 0,
                                              "latent": "K0", "latent_t": 1}]}]]

print("ONE encode call, batch axis as time — not one call per frame")
v = VAE()
out, n, info = R(mc.go(COND, v, 141, context_images=Img(60), context_frames="22"))
check("one call", len(v.calls), 1)
check("and it got all 22 frames at once", v.calls[0], 22)
check("frames reported", n, 22)

print("steps land at their own pixel offsets, per FRAME_PER_TOKEN (1,4,4,4,4)")
rows = out[0][1]["minimax_keyframes"]
ctx = [r["resolved_frame_index"] for r in rows if r["latent"] != "K0"]
check("22 frames -> 7 steps", len(ctx), 7)
check("offsets", ctx, [0, 1, 5, 9, 13, 17, 18])

print("the grid: only 39/22/5/1 are distinct, and off-grid snaps DOWN")
for asked, want in (("39", 39), ("22", 22), ("5", 5), ("1", 1)):
    _, got, _ = R(mc.go(COND, VAE(), 141, context_images=Img(60), context_frames=asked))
    check(f"{asked} -> {want}", got, want)
_, got, _ = R(mc.go(COND, VAE(), 141, context_images=Img(30), context_frames="39"))
check("only 30 frames available -> 22, not a short 39", got, 22)

print("coexistence — anchors and motion context are not exclusive")
check("the existing keyframe survives",
      sum(1 for r in rows if r["latent"] == "K0"), 1)
lat = out[0][1]["cond_video_latents"]
check("references come first", lat[:2], ["R1", "R2"])
check("then every keyframe row", len(lat), 2 + 1 + 7)
check("the input conditioning was not mutated",
      len(COND[0][1]["minimax_keyframes"]), 1)

print("passthrough")
check("no context wired", R(mc.go(COND, VAE(), 141))[1], 0)
check("context_frames 0", R(mc.go(COND, VAE(), 141, Img(60), "0"))[1], 0)
try:
    mc.go(COND, VAE(), 39, context_images=Img(60), context_frames="39")
    check("context filling the chunk raises", "no error", "ValueError")
except ValueError as e:
    check("context filling the chunk names the cause", "do not fit" in str(e), True)

print()
print("FAIL" if fails else "motion context: all checks pass")
sys.exit(1 if fails else 0)
