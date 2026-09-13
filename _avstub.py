"""A torch stub with real values, for testing the AV latent mask nodes.

NOT A TEST, and not imported by anything ComfyUI loads. `H3LatentBracket` and
`H3LatentInsert` are index arithmetic over two tensors -- no kernels, no
gradients -- and every way of getting that arithmetic wrong is SILENT: a mask
that holds too much renders a clip that ignores the prompt, one that holds too
little renders a clip that ignores the source, and both come back as "the model
did something odd" rather than as an error.

So the stub RECORDS writes and keeps values, which is what the assertions read.
It lives here rather than in one test file because the insert node's contract is
"the bracket, with the tail moved", and checking that against a SECOND hand-rolled
array implementation would be checking two stubs against each other.

Only what the nodes touch: clone, ones_like, linspace, cos, and slice assignment
on axis 2 of the video tensor and the last axis of the audio one.
"""
import importlib.util
import math
import pathlib
import sys
import types


class T:
    """A dense float array with numpy-ish slicing on one moving axis."""

    def __init__(self, shape, fill=0.0, data=None):
        self.shape = tuple(shape)
        n = 1
        for s in self.shape:
            n *= s
        self.data = list(data) if data is not None else [float(fill)] * n
        self.device = "cpu"
        self.dtype = "float32"

    # the nodes only ever index [:, :, a:b] on video and [..., a:b] on audio, so
    # the last axis (audio) and axis 2 (video) are the only ones that move.
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

    def _strides(self, ax):
        stride = 1
        for s in self.shape[ax + 1:]:
            stride *= s
        outer = 1
        for s in self.shape[:ax]:
            outer *= s
        return outer, stride

    def __setitem__(self, key, value):
        ax, span = self._span(key)
        vals = value.data if isinstance(value, T) else None
        outer, stride = self._strides(ax)
        w = 0
        for o in range(outer):
            for i in span:
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
        outer, stride = self._strides(ax)
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
        outer, stride = self._strides(axis)
        out = []
        for o in range(outer):
            base = (o * self.shape[axis] + axis_index) * stride
            out.extend(self.data[base:base + stride])
        return out


def install():
    """Put the stub modules in sys.modules and return the loaded `mask` module."""
    _t = types.ModuleType("torch")
    _t.Tensor = T
    _t.ones_like = lambda x: T(x.shape, 1.0)
    # H3LatentInsert allocates its own canvas rather than being handed one
    _t.zeros = lambda shape, device=None, dtype=None: T(shape, 0.0)
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

    root = pathlib.Path(__file__).resolve().parent
    pkg = types.ModuleType("h3b")
    pkg.__path__ = [str(root)]
    sys.modules["h3b"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"h3b.{name}", root / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"h3b.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    # avlatent.av unpacks the nested pair; stub it to match the stub NestedTensor
    _av = types.ModuleType("h3b.avlatent")
    _av.av = lambda s: (s[0], s[1])
    sys.modules["h3b.avlatent"] = _av
    _geom = types.ModuleType("h3b.geometry")
    _geom.cover_crop = _geom.crop_to_multiple = lambda *a, **k: None
    sys.modules["h3b.geometry"] = _geom
    _cp = types.ModuleType("h3b.chunkplan")
    _cp.snap_context = lambda n: int(n)
    sys.modules["h3b.chunkplan"] = _cp
    load("timing")
    return load("mask")


def latent_t(frames):
    """Core's `video_latent_t`, repeated so a test can build a clip of the
    right shape without importing ComfyUI."""
    return 2 if frames <= 5 else (int(frames) - 5) // 17 * 5 + 2
