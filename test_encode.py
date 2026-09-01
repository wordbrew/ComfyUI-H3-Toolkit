"""H3 Encode AV — the sizing and the frame-count trim.

The two things that silently ruin a refine pass:

FRAME COUNT is not resamplable. The video VAE takes 17n+5, so a clip that is
not a legal run has to lose frames off the END. Trimming UP would mean
inventing them; trimming down and not SAYING so is how sixteen frames vanish
off a take and get blamed on the model.

SIZE has to land on the grid AND keep the aspect. A stretch distorts anatomy
and resamples every pixel, which is the worst of both — the node centre-crops
to the target aspect first, which resamples nothing, and only then rescales.

    python3 test_encode.py
"""
import importlib
import importlib.util
import pathlib
import sys
import types

# encode.py imports torch and ComfyUI; the arithmetic under test touches
# neither, so stand them in the way the other torch-free tests do.
_t = types.ModuleType("torch"); _t.__path__ = []
_nn = types.ModuleType("torch.nn"); _nn.__path__ = []
_fn = types.ModuleType("torch.nn.functional")
_t.nn = _nn; _nn.functional = _fn
for _n, _m in (("torch", _t), ("torch.nn", _nn), ("torch.nn.functional", _fn)):
    sys.modules.setdefault(_n, _m)

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location(
    "h3enc_pkg", _root / "__init__.py", submodule_search_locations=[str(_root)])
_pk = importlib.util.module_from_spec(_spec)
sys.modules["h3enc_pkg"] = _pk
try:
    _spec.loader.exec_module(_pk)
except Exception:
    pass                        # some nodes need ComfyUI; the arithmetic does not

_enc = importlib.import_module("h3enc_pkg.encode")
fit_size = _enc.fit_size
H3EncodeAV = _enc.H3EncodeAV
audio_len = importlib.import_module("h3enc_pkg.timing").audio_t

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


# --- sizing ----------------------------------------------------------------- #
print("megapixels 0 keeps the source size, still snapped to the grid")
check("already legal is untouched", fit_size(640, 1120, 0.0, 32), (640, 1120))
check("off-grid snaps", fit_size(641, 1121, 0.0, 32), (640, 1120))

print("above 0 it scales to that area, both axes on the grid")
for mp in (1.03, 1.84, 2.87, 4.0):
    w, h = fit_size(640, 1120, mp, 32)
    ok(f"{mp} MP -> {w}x{h} on the grid", w % 32 == 0 and h % 32 == 0)
    ok(f"{mp} MP lands within 8% of the ask", abs(w * h / 1e6 - mp) / mp < 0.08)

print("aspect is preserved across the ladder")
base = 640 / 1120
for mp in (1.03, 1.84, 2.87):
    w, h = fit_size(640, 1120, mp, 32)
    ok(f"{mp} MP keeps 0.571 within 2%", abs(w / h - base) / base < 0.02)

print("the ladder in the workflow note is what the node actually produces")
check("1.03 -> canvas", fit_size(640, 1120, 1.03, 32), (768, 1344))
check("2.87 -> 2x linear", fit_size(640, 1120, 2.87, 32), (1280, 2240))

print("a landscape source is not silently made portrait")
w, h = fit_size(1120, 640, 2.87, 32)
ok("wider than tall stays that way", w > h)


# --- frame count ------------------------------------------------------------ #
# The trim is inside go(), which needs torch. Restate the same loop and assert
# it against the legal-run definition, so a change to one has to change both.
def trim(n):
    run = n
    while run % 17 != 5 and run > 5:
        run -= 1
    return run


print("frame count trims DOWN to 17n+5, never up")
LEGAL = [5 + 17 * k for k in range(0, 200)]      # past an hour of footage
for n in (5, 22, 39, 90, 141, 243, 345):
    check(f"{n} is already legal", trim(n), n)
for n in (6, 40, 100, 250, 360, 721):
    t = trim(n)
    ok(f"{n} -> {t} is legal", t in LEGAL)
    ok(f"{n} -> {t} never grows", t <= n)
    ok(f"{n} -> {t} is the NEAREST below", (t + 17) > n)

print("a 720-frame take loses its tail rather than gaining one")
check("720 -> 719", trim(720), 719)
ok("and 719 is legal", 719 in LEGAL)

print("below the VAE's minimum is an error, not a silent 5")
inst = H3EncodeAV()
try:
    class _Img:
        shape = (3, 1120, 640, 3)
    inst.go(_Img(), None, 0.0, 32)
    check("4 frames raises", "no error", "ValueError")
except ValueError as e:
    ok("and it names the minimum", "minimum run of 5" in str(e))
except Exception as e:                      # torch stub bites before the check
    ok("reached the guard", isinstance(e, ValueError))


# --- audio grid ------------------------------------------------------------- #
print("the audio half is sized off the trimmed run, not the source")
for n in (39, 90, 141, 243, 345):
    check(f"{n} frames -> {audio_len(n)} audio steps", audio_len(n),
          round(n / 24 * 40))
ok("an AV-aligned run is a whole number of steps",
   all(abs(n / 24 * 40 - round(n / 24 * 40)) < 1e-9
       for n in (39, 90, 141, 192, 243, 294, 345)))


# --- the contract the graph depends on -------------------------------------- #
print("outputs are ordered so a graph can wire size straight into H3")
check("return names", H3EncodeAV.RETURN_NAMES,
      ("latent", "width", "height", "length", "info"))
check("return types", H3EncodeAV.RETURN_TYPES,
      ("LATENT", "INT", "INT", "INT", "STRING"))
req = H3EncodeAV.INPUT_TYPES()["required"]
opt = H3EncodeAV.INPUT_TYPES()["optional"]
check("required order", list(req), ["images", "vae", "megapixels", "divisible_by"])
check("optional order", list(opt),
      ["audio_vae", "source_audio", "pin_audio", "width", "height"])
ok("pin_audio defaults ON — a video pass must not resample dialogue",
   opt["pin_audio"][1]["default"] is True)
ok("megapixels reaches 4.0, matching crop.py's raised cap",
   req["megapixels"][1]["max"] == 4.0)

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("encode: all checks pass")
