"""H3LatentPin's phase arithmetic and its guards, with a stub torch."""
import sys, types

class T:
    """Enough tensor to slice, clone and assign into."""
    def __init__(self, shape, fill=0.0, tag="t"):
        self.shape = tuple(shape); self.fill = fill; self.tag = tag
        self.writes = []
    def clone(self):
        c = T(self.shape, self.fill, self.tag); c.writes = list(self.writes); return c
    def to(self, *a, **k): return self
    def __getitem__(self, key): return T(self.shape, self.fill, f"{self.tag}[{key}]")
    def __setitem__(self, key, val): self.writes.append((key, getattr(val, "tag", val)))
    @property
    def device(self): return "cpu"
    @property
    def dtype(self): return "f32"
    # the audio feather does arithmetic on the ramp; the VALUES do not matter
    # here, only that the slicing and the guards are right
    def __mul__(self, o): return self
    __rmul__ = __sub__ = __rsub__ = __add__ = __radd__ = __mul__

torch = types.ModuleType("torch")
torch.ones_like = lambda x: T(x.shape, 1.0, "ones")
torch.linspace = lambda a, b, n, device=None, dtype=None: T((n,), 0.0, "ramp")
torch.cos = lambda x: x
nn = types.ModuleType("torch.nn"); nn.functional = types.ModuleType("torch.nn.functional")
torch.nn = nn; torch.__path__ = []; nn.__path__ = []
for n, m in (("torch", torch), ("torch.nn", nn), ("torch.nn.functional", nn.functional)):
    sys.modules.setdefault(n, m)
nt = types.ModuleType("comfy.nested_tensor")
nt.NestedTensor = lambda pair: ("nested", pair)
comfy = types.ModuleType("comfy"); comfy.nested_tensor = nt; comfy.__path__ = []
sys.modules.setdefault("comfy", comfy); sys.modules.setdefault("comfy.nested_tensor", nt)

import importlib.util, pathlib
root = pathlib.Path("/home/cgree/projects/ComfyUI-H3-Toolkit")
spec = importlib.util.spec_from_file_location("h3p", root / "__init__.py",
                                              submodule_search_locations=[str(root)])
pk = importlib.util.module_from_spec(spec); sys.modules["h3p"] = pk
try: spec.loader.exec_module(pk)
except Exception: pass
import importlib
mask = importlib.import_module("h3p.mask")

fails = []
def check(l, got, want):
    if got != want: fails.append(f"{l}: got {got!r}, want {want!r}"); print("  FAIL", l, got, want)
    else: print("  ok  ", l)

def latent(vt, at, h=70, w=40):
    return {"samples": ("nested", (T((1, 24, vt, h, w), 0.0, "vid"),
                                   T((1, 32, 2, at), 0.0, "aud")))}

# av() unpacks the nested pair; feed it what it expects
mask.av = lambda s: s[1]
pin = mask.H3LatentPin()

# 243-frame chunk = 72 video steps / 405 audio steps; a 39-frame pin is 12 / 65
out = pin.go(latent(72, 405), latent(72, 405), "39", 1.0, 8)["result"]
check("pinned frames reported", out[1], 39)
check("12 video and 65 audio steps named in info",
      "12 video / 65 audio" in out[2], True)

# the grid: 17n+5 pixel frames -> 5n+2 latent steps
for frames, steps in ((5, 2), (22, 7), (39, 12)):
    o = pin.go(latent(72, 405), latent(72, 405), str(frames), 1.0, 0)["result"]
    check(f"{frames} frames -> {steps} video steps",
          f"= {steps} video" in o[2], True)

# an off-grid request snaps DOWN to the grid rather than slicing at a phase the
# prefix does not have -- this is the bug that made this node splice
o = pin.go(latent(72, 405), latent(72, 405), "30", 1.0, 0)["result"]
check("30 snaps to 22, not sliced as 30", o[1], 22)

check("0 passes the latent straight through",
      pin.go(latent(72, 405), latent(72, 405), "0", 1.0, 0)[1], 0)

# guards, all of which used to be silent corruption
for label, tgt, prev, err in (
    ("a pin longer than the target", latent(7, 40), latent(72, 405), "does not fit"),
    ("a previous clip shorter than the pin", latent(72, 405), latent(7, 40), "shorter than the pin"),
    ("a resolution change", latent(72, 405), latent(72, 405, h=44, w=40), "one resolution"),
):
    try:
        pin.go(tgt, prev, "39", 1.0, 0)
        check(label + " raises", "no error", "ValueError")
    except ValueError as exc:
        check(label + " names the cause", err in str(exc), True)

print()
print("FAIL" if fails else "H3LatentPin: all checks pass")
sys.exit(1 if fails else 0)
