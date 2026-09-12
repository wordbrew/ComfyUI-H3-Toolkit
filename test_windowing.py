"""Window scheduling arithmetic. No torch, no ComfyUI.

`window_schedule` is a REPLICA of comfy.context_windows.create_windows_static_
standard, kept in this pack so a plan can be read without importing comfy (which
initialises CUDA). A replica that drifts from the original is worse than no
replica, so the cases here are the ones that pin its behaviour: the clamp, the
stride, and the frame grid.

    python3 test_windowing.py
"""

import pathlib
import sys
import types

sys.modules.setdefault("torch", types.ModuleType("torch"))

from timing import video_latent_t                        # noqa: E402
from windowing import (FRAME_PER_TOKEN, audio_ticks_for_frames,      # noqa: E402
                       crop_guide, frames_for_latent, map_modalities,
                       window_schedule)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, cond):
    if not cond:
        fails.append(label)


# --- the latent grid round-trips ------------------------------------------- #
# legal runs are 17n+5 and legal latent counts are 5n+2; the two must agree in
# both directions or a window lands between frames
for f in (5, 22, 39, 56, 90, 141, 192, 243, 294, 345, 396, 447):
    check(f"round trip {f}", frames_for_latent(video_latent_t(f)), f)
check("latent 2 is the floor", frames_for_latent(2), 5)
check("192 frames", video_latent_t(192), 57)
check("345 frames", video_latent_t(345), 102)

# --- the schedule ---------------------------------------------------------- #
# 345 frames at window 141 / overlap 39 divides exactly: three windows, each
# starting a full stride after the last, no clamp
sched = window_schedule(102, 42, 12)
check("345/141/39 is three clean windows", sched, [(0, 42), (30, 72), (60, 102)])
ok("and every overlap is the one asked for",
   all(sched[i - 1][1] - sched[i][0] == 12 for i in range(1, len(sched))))

# 192 frames at the same settings does NOT divide. The last window is pulled
# BACK to fit, so it shares 27 latent frames rather than 12 -- silently a
# gentler run than the settings say, and not comparable to one that divides.
sched = window_schedule(57, 42, 12)
check("192/141/39 clamps to two", sched, [(0, 42), (15, 57)])
check("and the real overlap is bigger than asked", sched[0][1] - sched[1][0], 27)

# a clip shorter than one window is a single window, not a negative start
check("short clip", window_schedule(20, 42, 12), [(0, 42)])
ok("never starts before zero",
   all(a >= 0 for a, _ in window_schedule(30, 42, 12)))

# the 90-frame window the earlier measurements used
check("192/90/39", window_schedule(57, 27, 12), [(0, 27), (15, 42), (30, 57)])

# --- the audio grid -------------------------------------------------------- #
# every window must start on an exact 40 Hz tick, so the PIXEL stride has to
# divide by 3. At a 90 window that leaves exactly one legal overlap.
def stride_frames(wf, of):
    d = video_latent_t(wf) - video_latent_t(of)
    return frames_for_latent(d + 2) - 5


check("90/39 stride", stride_frames(90, 39), 51)
check("90/22 stride", stride_frames(90, 22), 68)
check("90/5 stride", stride_frames(90, 5), 85)
check("141/39 stride", stride_frames(141, 39), 102)
legal = [o for o in range(5, 90, 17) if stride_frames(90, o) % 3 == 0]
check("39 is the only both-clocks overlap at a 90 window", legal, [39])
ok("102 divides by 3, so 141/39 is on both clocks",
   stride_frames(141, 39) % 3 == 0)

# --- the frame phase ------------------------------------------------------- #
# FRAME_PER_TOKEN is (1,4,4,4,4), so a stride that is not a multiple of 5 latent
# frames puts every window on a different intra-window frame grid
for wf, of in ((90, 39), (90, 22), (141, 39), (192, 39)):
    d = video_latent_t(wf) - video_latent_t(of)
    ok(f"{wf}/{of} latent stride is a multiple of 5", d % 5 == 0)

# --- LongFormLayout's build order, and its signature ----------------------- #
# Two bugs, both invisible offline because video.py imports comfy.
#
# `_offset_target` walks self.segments, so it cannot run before the segment
# table is assigned. It did, on the keyframe branch only, and every windowed
# render with a keyframe raised AttributeError.
#
# And the signature must stay a 5-TUPLE. `_forward` reuses the prebuilt layout
# only when the signature matches what it computes; appending window_start makes
# every offset window miss, rebuild without the offset (already consumed), and
# silently revert to origin positioning -- right on window 0, wrong on the rest.
src = (pathlib.Path(__file__).resolve().parent / "video.py").read_text(encoding="utf-8")
ok("_offset_target runs after self.segments on the keyframe branch",
   src.rindex("self._offset_target(") > src.index("self.segments = seg"))
ok("both branches offset the target", src.count("self._offset_target(") >= 2)
ok("the layout signature stays a 5-tuple",
   "self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)" in src
   and "+ (int(window_start),)" not in src)

# --- the subclasses against an UNPATCHED core ------------------------------ #
# H3WindowingState and H3ContextHandler override core methods by NAME. A typo
# means the override is never called and core's version runs -- no error, a bad
# render. That has happened twice on this code. So: every method they define has
# to exist on the core base class under that name, with a compatible signature.
#
# Read core's SOURCE rather than importing it. Importing comfy.context_windows
# initialises CUDA, and this file is meant to run with no torch at all. Prefer
# the git HEAD version, because the working tree may still carry the old core
# patch and the whole point is that these work WITHOUT it.
import ast                                                      # noqa: E402
import os                                                       # noqa: E402
import subprocess                                               # noqa: E402

# name -> the core class it subclasses
SUBCLASSES = {"H3WindowingState": "WindowingState",
              "H3ContextHandler": "IndexListContextHandler"}
# deliberately new; core has no equivalent to check against
# methods that are OURS and override nothing. Listed one by one on purpose: an
# "ignore anything private" rule would have hidden _widen_for_margin sitting on
# the wrong class, which is the exact mistake this check exists to catch.
NEW_METHODS = {"_get_modality_dims", "__post_init__", "_widen_for_margin",
               "_report_layout_builds"}
# module-level function -> the core function it mirrors
MIRRORED_FUNCS = {"_h3_sampler_sample_wrapper": "_sampler_sample_wrapper"}


def comfy_root():
    for p in (os.environ.get("COMFYUI_PATH"),
              pathlib.Path(__file__).resolve().parents[2],
              "/mnt/c/SD/ComfyUI/Comfy-03-15-2026/ComfyUI"):
        if not p:
            continue
        root = pathlib.Path(p)
        if (root / "comfy" / "context_windows.py").is_file():
            return root
    return None


def core_source(root):
    """core's context_windows.py at git HEAD, or the working tree if not a repo."""
    r = subprocess.run(["git", "-C", str(root), "show",
                        "HEAD:comfy/context_windows.py"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout:
        return r.stdout, "git HEAD"
    return (root / "comfy" / "context_windows.py").read_text(encoding="utf-8"), \
        "working tree (may be patched)"


def classes_in(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def methods_of(cls_node):
    return {n.name: n for n in cls_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def signature(fn):
    """(positional names, how many of them have defaults, *args?, **kwargs?)"""
    a = fn.args
    names = [x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
    return names, len(a.defaults), a.vararg is not None, a.kwarg is not None


def compare(label, mine, theirs):
    """Core's parameters must be a prefix of ours, and any extra must be optional."""
    m_names, m_defaults, m_star, m_kw = signature(mine)
    t_names, _t_defaults, t_star, t_kw = signature(theirs)
    if m_names[:len(t_names)] != t_names:
        fails.append(f"{label} signature does not match core's:\n"
                     f"      core  {t_names}\n      ours  {m_names}")
        return
    extra = m_names[len(t_names):]
    if extra and len(m_names) - m_defaults > len(t_names):
        fails.append(f"{label} adds required parameter(s) {extra}; core calls it "
                     f"with {len(t_names)} argument(s), so they need defaults")
    if (t_star and not m_star) or (t_kw and not m_kw):
        fails.append(f"{label} drops core's *args/**kwargs")


root = comfy_root()
if root is None:
    print("windowing: SKIPPED the core-contract check — no ComfyUI found. Set "
          "COMFYUI_PATH to run it.")
else:
    core_text, origin = core_source(root)
    core = classes_in(ast.parse(core_text))
    core_funcs = {n.name: n for n in ast.parse(core_text).body
                  if isinstance(n, ast.FunctionDef)}
    ours_tree = ast.parse(
        (pathlib.Path(__file__).resolve().parent / "windowing.py")
        .read_text(encoding="utf-8"))
    ours = classes_in(ours_tree)
    our_funcs = {n.name: n for n in ast.walk(ours_tree)
                 if isinstance(n, ast.FunctionDef)}

    checked = 0
    for name, base in SUBCLASSES.items():
        if name not in ours:
            fails.append(f"windowing.py defines no class {name}")
            continue
        if base not in core:
            fails.append(f"core has no class {base} for {name} to subclass")
            continue
        bases = [b.id for b in ours[name].bases if isinstance(b, ast.Name)]
        if base not in bases:
            fails.append(f"{name} subclasses {bases}, not {base}")
        core_methods = methods_of(core[base])
        for m_name, m_node in methods_of(ours[name]).items():
            if m_name in NEW_METHODS:
                continue
            if m_name not in core_methods:
                fails.append(f"{name}.{m_name} overrides nothing — core's {base} "
                             f"has no {m_name}, so it would never be called")
                continue
            compare(f"{name}.{m_name}", m_node, core_methods[m_name])
            checked += 1

    for ours_name, core_name in MIRRORED_FUNCS.items():
        if ours_name not in our_funcs:
            fails.append(f"windowing.py defines no {ours_name}")
        elif core_name not in core_funcs:
            fails.append(f"core has no {core_name} for {ours_name} to mirror")
        else:
            compare(ours_name, our_funcs[ours_name], core_funcs[core_name])
            checked += 1

    # the node has to install the handler itself; resolving core's node would
    # put core's handler back, which is the whole bug
    src = (pathlib.Path(__file__).resolve().parent / "windowing.py").read_text(
        encoding="utf-8")
    ok("H3ContextWindows no longer calls core's ContextWindowsManual",
       "ContextWindowsManual\"" not in src and "ContextWindowsManualNode" not in src)
    ok("the handler goes into model_options itself",
       'model_options["context_handler"]' in src)

    print(f"windowing: {checked} override(s) checked against core "
          f"({origin}) at {root}")

# --- the audio window's LENGTH ---------------------------------------------- #
# A window of N pixel frames is worth round(N * 40/24) ticks. Unioning each
# token's span with a floored start and a ceiled end returned one MORE than that
# on any start off the 51-frame grid -- 8 of 13 starts for a 90-frame window. It
# never fired at the defaults (window 90 / overlap 39 is a stride of 51), which
# is exactly why it survived: a default is not a guarantee.
print("every window gets the tick count its frame count is worth")
_shapes = [[1, 24, 102, 70, 40], [1, 32, 2, 567]]      # a 345-frame clip
_bad = []
for _s in range(0, 102 - 27 + 1, 5):                    # every legal 90f window
    _idx = list(range(_s, _s + 27))
    _got = len(map_modalities(_idx, _shapes, 2)[1])
    _want = audio_ticks_for_frames(sum(FRAME_PER_TOKEN[v % 5] for v in _idx))
    if _got != _want:
        _bad.append((_s, _got, _want))
check("no window is over or under length", _bad, [])
check("a 90-frame window is 150 ticks",
      len(map_modalities(list(range(0, 27)), _shapes, 2)[1]), 150)
check("a start off the 51-grid is still 150",
      len(map_modalities(list(range(5, 32)), _shapes, 2)[1]), 150)
check("the last window ends exactly at the clip",
      map_modalities(list(range(75, 102)), _shapes, 2)[1][-1] + 1, 567)
check("3 frames is 5 ticks", audio_ticks_for_frames(3), 5)
check("141 frames is a whole 235 ticks", audio_ticks_for_frames(141), 235)

# --- margin windows --------------------------------------------------------- #
# Evaluate wider than you fuse. A window's edge tokens only ever saw context on
# one side, and that is where neighbouring windows disagree most -- blending
# cannot fix a disagreement about CONTENT, which is where the August work
# stopped. The margin throws those tokens away instead of averaging them.
#
# Two invariants have to hold or the model sees a clip shape it was not trained
# on: every widened window is still 5j+2 tokens starting on a 5-token boundary,
# and the inner spans still tile the clip exactly once.
def _widen(windows, total, m):
    out = []
    for idx in windows:
        s, L = idx[0], len(idx)
        lo, hi = max(0, s - m), min(total, s + L + m)
        out.append((list(range(lo, hi)), s - lo, L))
    return out


print("a margin widens the evaluation without breaking the token phase")
_TOTAL, _W, _S, _M = 102, 27, 15, 5          # 345f clip, 90f window, 51f stride
_starts = list(range(0, _TOTAL - _W + 1, _S))
if _starts[-1] != _TOTAL - _W:
    _starts.append(_TOTAL - _W)
_inner = [list(range(s, s + _W)) for s in _starts]
_wide = _widen(_inner, _TOTAL, _M)
check("every widened window starts on a 5-token boundary",
      [w[0][0] % 5 for w in _wide], [0] * len(_wide))
check("and is still a legal 5j+2 length",
      [len(w[0]) % 5 for w in _wide], [2] * len(_wide))
check("the inner span is the original window",
      [w[2] for w in _wide], [_W] * len(_wide))
_cov = {}
for _idx, _off, _n in _wide:
    for _k in _idx[_off:_off + _n]:
        _cov[_k] = _cov.get(_k, 0) + 1
check("no frame is left unfused",
      [k for k in range(_TOTAL) if k not in _cov], [])
ok("the clip's first and last windows take a one-sided margin",
   len(_wide[0][0]) < len(_wide[1][0]) and len(_wide[-1][0]) < len(_wide[1][0]))
check("a margin of 0 changes nothing",
      [len(w[0]) for w in _widen(_inner, _TOTAL, 0)], [_W] * len(_inner))

# --- guide clips are cropped to the window, not kept or dropped whole -------- #
# A still is described entirely by its start index, so keep-or-drop was right for
# one. A multi-frame AddGuide clip is not: one starting inside a window and
# running past its end used to be kept WHOLE, and one starting before the window
# and running into it was dropped, losing guidance the window needed.
class _Z:
    """Stands in for a [1, 24, T, h, w] guide latent."""

    def __init__(self, t, tag="z"):
        self.t, self.tag = t, tag

    @property
    def shape(self):
        return (1, 24, self.t, 70, 40)

    def dim(self):
        return 5

    def __getitem__(self, key):
        sl = key[2]
        return _Z(sl.stop - sl.start, f"{self.tag}[{sl.start}:{sl.stop}]")


print("a guide clip is cropped to the window, on whole token cycles")
_still = {"resolved_frame_index": 30, "latent": _Z(1)}
check("a still inside the window is rebased",
      crop_guide(_still, 17, 107)["resolved_frame_index"], 13)
check("a still outside it is dropped", crop_guide(_still, 120, 210), None)

# 27 steps = 90 frames
_clip = {"resolved_frame_index": 0, "latent": _Z(27, "G")}
_c = crop_guide(_clip, 51, 141)
ok("a clip running into the window is KEPT, not dropped", _c is not None)
check("its tail is the part that survives", _c["latent"].t, 12)
check("cropped to a legal 5j+2 length", _c["latent"].t % 5, 2)
check("and rebased to window-local frames", _c["resolved_frame_index"], 0)
_whole = crop_guide(_clip, 0, 90)
check("a clip the window fully contains is untouched", _whole["latent"].t, 27)
check("a clip the window never reaches is dropped",
      crop_guide({"resolved_frame_index": 0, "latent": _Z(12)}, 200, 290), None)

# the crop must never hand back a length the model cannot read
_bad = []
for _ws in range(0, 200, 17):
    _r = crop_guide({"resolved_frame_index": 0, "latent": _Z(52, "L")}, _ws, _ws + 90)
    if _r is None:
        continue
    if _r["latent"].t % 5 != 2 or _r["resolved_frame_index"] < 0:
        _bad.append((_ws, _r["latent"].t, _r["resolved_frame_index"]))
check("every crop is a legal clip at a non-negative local frame", _bad, [])

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("windowing: all checks pass")
