# Patches to other people's code

Fixes to third-party packs and to ComfyUI core that this pack's workflows need.
They live here rather than in the install because a Manager update reverts a
custom node folder with `git pull` or a hard reset, and the patch goes silently.
Nothing here is applied automatically.

After any Manager update, check whether a patch survived:

    git -C "<pack folder>" diff --stat

Nothing printed means it went. Reapply with the command under each heading.

---

## depthanythingv2-contiguous.patch

**Pack:** [ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2),
clean at `5531878`.
**Symptom:** `DepthAnything_V2` raises
`RuntimeError: unsupported operation: more than one element of the written-to
tensor refers to a single memory location`.

**Cause.** `nodes.py` builds its three-channel output with `expand()`, which
returns a view whose channels all alias one memory location, then runs
`sub_`/`div_`/`clamp_` on it in place. What normally saves it is the
`F.interpolate` a few lines down, which materialises a real tensor — but that
only runs when the depth model's output size differs from the input's. The model
crops its input to a multiple of 14, so the sizes match, the branch is skipped
and the in-place op raises **whenever the input's height and width are BOTH
multiples of 14**.

That is easy to hit here. `H3MatchSource` conforms a clip to a megapixel target
and `H3SubjectCrop` rounds to 32, so a 4:3 source at 0.62 MP lands on 896x672 —
64 and 48 fourteenths. Halving it first (the usual `ImageScaleBy 0.5` before a
depth pass) does not help: 448x336 is still both. A 1.27:1 source gives 896x704,
352 is not a multiple of 14, and the bug stays hidden. So it looks like a clip
that "suddenly" broke a workflow that had always worked.

**Fix:** `.contiguous()` after the `expand()`. Correct at every resolution.

    cd "C:\SD\ComfyUI\Comfy-03-15-2026\ComfyUI\custom_nodes\ComfyUI-DepthAnythingV2"
    git apply "<this repo>/patches/depthanythingv2-contiguous.patch"

Revert with `git checkout -- nodes.py`.

**Workflow-only alternative,** if you would rather not patch: set the
`ImageScaleBy` before the depth node to a factor that lands off the 14 grid —
0.60 gives 538x403 on this clip. It is luck rather than a fix; it depends on the
source's aspect and another clip can land back on the grid.

**THE PACK'S OWN LAYOUT PATCH HAS TO AGREE WITH THIS ONE.** `video.py`
replaces `PackedLayout` with a `LongFormLayout` subclass so references and
keyframes can share one packed sequence. When core started passing
`window_start`, that subclass did not accept it and every windowed render
through the long-form conditioning node died with

    TypeError: LongFormLayout.__init__() got an unexpected keyword argument
    'window_start'

It forwards it now, but only when the running build's `PackedLayout` actually
takes it — the same introspection it already uses for `frame_count`, which core
dropped in 0.34.0 — so the pack runs against a patched and an unpatched core
alike. Its keyframe path applies the offset to the target grids too, and its
signature matches core's so the rebuild check behaves. Any future change to this
patch has to be mirrored there.

**Not reported upstream yet.**

---

## h3-window-absolute-positions.patch

**File:** `comfy/ldm/minimax/model.py` (ComfyUI core).
**Symptom:** a context-windowed H3 render is clean through the first window and
then flickers for the rest of the clip, the picture pulsing at the window period.

**Cause.** `PackedLayout` positions the target video and audio at `cursor`,
which is `text_len` plus the reference spans. That value does not depend on the
context window, so EVERY window's target is placed at the clip origin. The model
is told each window is the start of the shot, so it renders the start of the
shot, and the overlaps then crossfade between two openings.

It is the same fact motion context turns on: the rows are identical between a
reference and a keyframe, and only the TIME COORDINATES say "separate clip"
versus "this clip, earlier". Windowing said "separate clip" every time.

**Fix.** `PackedLayout` takes a `window_start` in pixel frames and offsets the
target grids by `FRAME_RESCALE * window_start` — the same unit and rate the
keyframe anchors already use a few lines above it. `window_start` joins the
layout signature, so two windows of the same SHAPE at different clip positions
cannot share a cached layout. `window_start=0` reproduces the old behaviour
exactly, which is what makes the toolkit-side toggle a real A/B.

The toolkit supplies the value: `H3ContextWindows` has an
`absolute_window_positions` widget, and the hook in `windowing.py` writes
`window_start_frames` into the payload per window. With the toggle off nothing
changes, so the patch is safe to leave applied.

    cd "C:\SD\ComfyUI\Comfy-03-15-2026\ComfyUI"
    git apply "<this repo>/patches/h3-window-absolute-positions.patch"

Revert with `git checkout -- comfy/ldm/minimax/model.py`.

**Measured 2026-08-28**, 192 frames, window 90 / overlap 39, three windows:
before, clean through window 0 then flickering; after, no flicker and motion
smooth throughout. Background content still shifts between windows — the overlap
blend is the only thing carrying content across a seam, and that is a separate
problem from where the window thinks it is.

**Two settings that have to be right or the patch cannot be judged**, both found
the hard way on the way to this:

- `schedule` must be `standard_static`. A uniform schedule re-derives every
  window position on every step (`pad = round(num_frames * ordered_halving(step))`),
  so no frame is rendered by a window in a consistent place — and with absolute
  positions on, each frame is told a different time on every step.
- `causal_window_fix` must be OFF. Core prepends an anchor frame to every window
  after the first, which makes them 28 latent frames where the first is 27. 28 is
  not on H3's 5n+2 latent grid: 27 latent is exactly 90 pixel frames, 28 is 94,
  which is not a legal run at all.

**Not reported upstream yet.**
