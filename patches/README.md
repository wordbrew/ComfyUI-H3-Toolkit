# Patches to other people's code

Fixes to third-party packs and to ComfyUI core that this pack's workflows need.
They live here rather than in the install because a Manager update reverts a
custom node folder with `git pull` or a hard reset, and the patch goes silently.
Nothing here is applied automatically.

After any ComfyUI or Manager update, run:

    python3 patches/apply.py            # what is applied, what is not
    python3 patches/apply.py --apply    # apply whatever is missing
    python3 patches/apply.py --revert   # take them all back out

It is idempotent -- an applied patch is reported and skipped, never applied
twice -- and it reports CONFLICT when core has moved under a patch, which means
the patch needs rebasing rather than forcing.

An update reverts these silently, and nothing errors afterwards -- the render
just comes out wrong. **No ComfyUI CORE file is patched any more:** both core
patches have been rewritten as subclasses that live in the pack, precisely
because a silent revert cost more than the patches saved. What is left is one
third-party pack.

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

## h3-modality-dim-context-windows.patch — SUPERSEDED, not needed

**Was:** `comfy/context_windows.py`, `comfy/model_base.py` (ComfyUI core).
**Symptom without it:** context windows on H3 sliced the audio modality on its
stereo-pair dimension instead of time, and the fuse died with
`size of tensor a (2) must match the size of tensor b (93)`.

Core's windowing assumes every modality's temporal axis sits at the primary's
`dim`. H3's video latent is `[B, 24, T, H/16, W/16]` -- time at dim 2 -- but its
audio latent is `[B, 32, 2, T]`, where dim 2 is the stereo pair and time is dim
3. LTXAV has time at dim 2 for both, so the assumption held until H3.

**Where it lives now:** `windowing.py`, as `H3WindowingState(WindowingState)`
and `H3ContextHandler(IndexListContextHandler)`. `H3ContextWindows` installs the
handler into `model.model_options["context_handler"]` directly instead of
calling core's `Context Windows (Manual)` node. The handler is only an object in
a dict, read back in `samplers.py`, so a subclass is the whole mechanism —
MMH3Tools' `nodes_windows.py` solves the same problem the same way, and this
follows its structure.

The model-side half of the patch is not replicated at all: it only added a
DEFAULT `context_modality_dim` to `BaseModel`, and `H3ContextHandler` reaches
for the hook with `getattr` and falls back to its own `dim` when a model has
none. `patch_h3_context_windows()` installs H3's implementation on the
`MiniMaxH3` class as before.

The patch file is kept for reference. Leaving it applied is harmless — the
subclass overrides every method it touched — but nothing needs it, and a
ComfyUI update reverting it now changes nothing.

## h3-window-absolute-positions.patch — SUPERSEDED, do not apply

The offset lives in the PACK now, in `video.py`'s `PackedLayout` subclass, which
shifts column 0 of the finished position table. Both position grids are affine in
the cursor --

    _video_t_grid(n, origin) = origin + cumsum(spans)
    _audio_grid(cursor, t)   = cursor + arange(t)

-- so adding a constant afterwards is arithmetically identical to starting from a
later cursor, and it needs no core edit at all. Verified 2026-08-30 on a stock
`comfy/ldm/minimax/model.py`: three windows, 40 layout builds each, every one
carrying its own offset.

Two things that had to be right, both of which bit first:

  THE SIGNATURE STAYS A 5-TUPLE. `_forward` reuses the prebuilt layout only when
  the signature matches what it computes. Appending window_start makes every
  offset window miss, rebuild without the offset, and revert to origin
  positioning -- right on window 0, wrong on all the rest.

  `H3EnableContextWindows` INSTALLS THE LAYOUT PATCH. It used to be installed
  only by this pack's own conditioning nodes, so a graph built on core's
  MiniMaxH3ReferenceToVideo never got it and every window stayed at the origin.

The patch file is kept only for reference. Applying it now would double the
offset.

## the old patch, for reference

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
