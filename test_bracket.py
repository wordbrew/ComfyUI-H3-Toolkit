"""H3LatentBracket holds both ends of a clip and generates the middle.

WHAT THIS FILE IS DEFENDING

  The node's whole contract is the noise mask it builds, and every way of
  getting that wrong is SILENT. A mask that holds too much renders a clip that
  ignores the prompt; a mask that holds too little renders one that ignores the
  source; both come back as "the model did something odd" rather than an error.

  The specific trap, and the reason this file exists: `x[..., -0:]` is the WHOLE
  tensor in Python, not an empty slice. A zero-length tail bracket written the
  obvious way holds the entire clip and generates nothing at all -- a render
  that costs full price and returns its own input.

  The arithmetic is checked against `timing`'s own helpers rather than repeated
  here, so a change to the frame grid fails in one place.

    python3 test_bracket.py
"""
import math
import sys
import types

# --- a torch stub, because this is index arithmetic and needs no kernels ----- #
# Only what the node touches: clone, ones_like, linspace, cos, and slice
# assignment. The stub RECORDS writes, which is what the assertions read.


class T:
    """A dense float array with numpy-ish slicing on the last two axes."""

    def __init__(self, shape, fill=0.0, data=None):
        self.shape = tuple(shape)
        n = 1
        for s in self.shape:
            n *= s
        self.data = list(data) if data is not None else [float(fill)] * n
        self.device = "cpu"
        self.dtype = "float32"

    # the node only ever indexes [:, :, a:b] on video and [..., a:b] on audio,
    # so the last axis (audio) and axis 2 (video) are the only ones that move.
    def _axis(self, key):
        if key is Ellipsis:
            return len(self.shape) - 1, slice(None)
        if isinstance(key, tuple):
            for i, k in enumerate(key):
                if isinstance(k, slice) and k != slice(None):
                    return (len(self.shape) - 1 if key[0] is Ellipsis else i), k
            return len(self.shape) - 1, slice(None)
        return 0, key

    def _span(self, key):
        ax, sl = self._axis(key)
        return ax, range(*sl.indices(self.shape[ax]))

    def __setitem__(self, key, value):
        ax, span = self._span(key)
        vals = value.data if isinstance(value, T) else None
        stride = 1
        for s in self.shape[ax + 1:]:
            stride *= s
        outer = 1
        for s in self.shape[:ax]:
            outer *= s
        w = 0
        for o in range(outer):
            for j, i in enumerate(span):
                base = (o * self.shape[ax] + i) * stride
                for k in range(stride):
                    if vals is None:
                        self.data[base + k] = float(value)
                    else:
                        self.data[base + k] = vals[w % len(vals)]
                        w += 1

    def __getitem__(self, key):
        ax, span = self._span(key)
        shape = list(self.shape)
        shape[ax] = len(span)
        stride = 1
        for s in self.shape[ax + 1:]:
            stride *= s
        outer = 1
        for s in self.shape[:ax]:
            outer *= s
        out = []
        for o in range(outer):
            for i in span:
                base = (o * self.shape[ax] + i) * stride
                out.extend(self.data[base:base + stride])
        return T(shape, data=out)

    # elementwise scalar arithmetic, which is all the feather ramp needs
    def _map(self, f):
        return T(self.shape, data=[f(v) for v in self.data])

    def __mul__(self, k):
        return self._map(lambda v: v * k)

    __rmul__ = __mul__

    def __add__(self, k):
        return self._map(lambda v: v + k)

    __radd__ = __add__

    def __sub__(self, k):
        return self._map(lambda v: v - k)

    def __rsub__(self, k):
        return self._map(lambda v: k - v)

    def clone(self):
        return T(self.shape, data=self.data)

    def to(self, *a, **k):
        return self

    def at(self, axis_index, axis=None):
        """Every value lying at one index of the moving axis."""
        axis = len(self.shape) - 1 if axis is None else axis
        stride = 1
        for s in self.shape[axis + 1:]:
            stride *= s
        outer = 1
        for s in self.shape[:axis]:
            outer *= s
        out = []
        for o in range(outer):
            base = (o * self.shape[axis] + axis_index) * stride
            out.extend(self.data[base:base + stride])
        return out


_t = types.ModuleType("torch")
_t.Tensor = T
_t.ones_like = lambda x: T(x.shape, 1.0)
_t.cos = lambda x: T(x.shape, data=[math.cos(v) for v in x.data])


def _linspace(a, b, n, device=None, dtype=None):
    if n == 1:
        return T((1,), data=[float(a)])
    step = (b - a) / (n - 1)
    return T((n,), data=[a + step * i for i in range(n)])


_t.linspace = _linspace
_fn = types.ModuleType("torch.nn.functional")
_nn = types.ModuleType("torch.nn")
_nn.functional = _fn
_t.nn = _nn
sys.modules.update({"torch": _t, "torch.nn": _nn, "torch.nn.functional": _fn})

_nested = types.ModuleType("comfy.nested_tensor")
_nested.NestedTensor = lambda pair: list(pair)
_comfy = types.ModuleType("comfy")
_comfy.nested_tensor = _nested
sys.modules.update({"comfy": _comfy, "comfy.nested_tensor": _nested})

import importlib.util  # noqa: E402
import pathlib  # noqa: E402

_root = pathlib.Path(__file__).resolve().parent
_pkg = types.ModuleType("h3b")
_pkg.__path__ = [str(_root)]
sys.modules["h3b"] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(f"h3b.{name}", _root / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"h3b.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# avlatent.av unpacks the nested pair; stub it to match the stub NestedTensor.
_av = types.ModuleType("h3b.avlatent")
_av.av = lambda s: (s[0], s[1])
sys.modules["h3b.avlatent"] = _av
_geom = types.ModuleType("h3b.geometry")
_geom.cover_crop = _geom.crop_to_multiple = lambda *a, **k: None
sys.modules["h3b.geometry"] = _geom
_cp = types.ModuleType("h3b.chunkplan")
_cp.snap_context = lambda n: int(n)
sys.modules["h3b.chunkplan"] = _cp
_load("timing")
mask = _load("mask")
from h3b.timing import audio_t, frame_groups  # noqa: E402

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)
        print(f"  FAIL {label}")


def run(total_frames, head, tail, feather=0, strength=1.0):
    t_v = 2 if total_frames <= 5 else (total_frames - 5) // 17 * 5 + 2
    t_a = audio_t(total_frames)
    src = [T((1, 24, t_v, 4, 4), 7.0), T((1, 32, 2, t_a), 9.0)]
    tgt = [T((1, 24, t_v, 4, 4), 0.0), T((1, 32, 2, t_a), 0.0)]
    node = mask.NODE_CLASS_MAPPINGS["H3LatentBracket"]()
    out = node.go({"samples": tgt}, {"samples": src}, head, tail, strength,
                  audio_feather_ticks=feather)
    lat = out["result"][0]
    return lat["samples"], lat["noise_mask"], out["result"][1], t_v, t_a


TOTAL = 345                      # 17*20+5, 102 video steps, 575 audio ticks
print("both ends are held and the middle is left to generate")
(sv, sa), (mv, ma), gen, t_v, t_a = run(TOTAL, 34, 34)
check("the clip is 102 video steps", t_v, 102)
check("34 frames is 10 video steps at each end",
      (mv.at(0, axis=2)[0], mv.at(9, axis=2)[0]), (0.0, 0.0))
check("step 10 is generated", mv.at(10, axis=2)[0], 1.0)
check("the last 10 steps are held",
      (mv.at(t_v - 10, axis=2)[0], mv.at(t_v - 1, axis=2)[0]), (0.0, 0.0))
check("the step before them is generated", mv.at(t_v - 11, axis=2)[0], 1.0)
check("held video carries the SOURCE, not the target",
      (sv.at(0, axis=2)[0], sv.at(t_v - 1, axis=2)[0]), (7.0, 7.0))
check("the middle is untouched target", sv.at(50, axis=2)[0], 0.0)
check("generated_frames is what is left", gen, sum(frame_groups(t_v)) - 68)

print("audio is held over the same real time")
check("34 frames is 57 ticks", audio_t(34), 57)
check("the first ticks are held", ma.at(0)[0], 0.0)
check("and the last", ma.at(t_a - 1)[0], 0.0)
check("the middle is generated", ma.at(t_a // 2)[0], 1.0)
check("held audio carries the source", sa.at(0)[0], 9.0)

# THE BUG THIS FILE WAS WRITTEN FOR. `x[..., -0:]` is the whole tensor, so a
# zero tail written the obvious way holds everything and generates nothing.
print("a zero-length bracket holds nothing, rather than everything")
(_, _), (mv0, ma0), _, t_v0, t_a0 = run(TOTAL, 34, 0)
check("the tail is free when tail_frames is 0", mv0.at(t_v0 - 1, axis=2)[0], 1.0)
check("and its audio is too", ma0.at(t_a0 - 1)[0], 1.0)
check("the head is still held", mv0.at(0, axis=2)[0], 0.0)
(_, _), (mv1, _), _, t_v1, _ = run(TOTAL, 0, 34)
check("the head is free when head_frames is 0", mv1.at(0, axis=2)[0], 1.0)
check("the tail is still held", mv1.at(t_v1 - 1, axis=2)[0], 0.0)

print("brackets snap to whole VAE chunks")
# 5 latent steps cover 17 pixel frames wherever they start, so a held region is
# always a round number of chunks and needs no phase arithmetic.
(_, _), (mv2, _), _, t_v2, _ = run(TOTAL, 40, 0)      # 40 -> 34
check("40 frames snaps down to 34, which is 10 steps",
      (mv2.at(9, axis=2)[0], mv2.at(10, axis=2)[0]), (0.0, 1.0))
(_, _), (mv3, _), _, _, _ = run(TOTAL, 16, 0)         # 16 -> 0
check("below one chunk nothing is held", mv3.at(0, axis=2)[0], 1.0)

print("the audio feather runs the right way at each end")
(_, _), (_, maf), _, _, t_af = run(TOTAL, 51, 51, feather=8)
h = audio_t(51)
ok("the head ramp rises into the generated middle",
   maf.at(h - 8)[0] < maf.at(h - 4)[0] < maf.at(h - 1)[0] < 1.0)
ok("the tail ramp falls into the held tail",
   maf.at(t_af - h)[0] > maf.at(t_af - h + 4)[0] > maf.at(t_af - h + 7)[0])
check("deep inside the head is still fully held", maf.at(0)[0], 0.0)
check("deep inside the tail is still fully held", maf.at(t_af - 1)[0], 0.0)
check("a feather of 0 leaves a hard edge",
      run(TOTAL, 51, 51, feather=0)[1][1].at(h - 1)[0], 0.0)

print("it refuses what it cannot do")
for label, args in (("holding the whole clip", (TOTAL, 340, 340)),
                    ("a head longer than the clip", (TOTAL, 3400, 0))):
    try:
        run(*args)
        fails.append(f"{label}: no error raised")
        print(f"  FAIL {label}: no error raised")
    except ValueError:
        pass

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("bracket: both ends held, the middle generated, and -0 does not mean all")
