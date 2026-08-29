# Context windows for H3

State as of 2026-08-28. What works, what it costs, and what it still cannot do.

## The short version

Context windows work on H3 **only with the absolute-positions patch applied and
two settings right**. Without the patch the picture flickers at the window
period from the second window onward. With it, motion is smooth throughout.

    schedule              standard_static      (NOT uniform)
    causal_window_fix     off
    window / overlap      both legal runs (17n+5); overlap 39 for AV work

Windowing and chunking cost about the same. They are not alternatives: they
produce different things.

## What was wrong, and why

`comfy/ldm/minimax/model.py`, `PackedLayout.__init__`, positions the target
video and audio at `cursor`:

```python
n_video = latent_t * frame_rows
pos.append(_video_grid(latent_t, frame, cursor))
```

`cursor` is `text_len` plus the reference spans. **It does not depend on the
window.** So every window's target rows are told they begin at the clip origin,
each window renders the opening of the shot, and the overlaps then crossfade
between two openings. That is the flicker.

It is the same fact motion context turns on: the rows the model sees are
identical between a reference and a keyframe, and only the TIME COORDINATES say
"separate clip" versus "this clip, earlier". Windowing said "separate clip"
every time.

The flicker starts *after* the first window because window 0 is also the only
one carrying a frame-0 keyframe — repeating it would snap the clip back to the
start image at every seam, so the toolkit drops it from later windows.

## The fix

`patches/h3-window-absolute-positions.patch` gives `PackedLayout` a
`window_start` in pixel frames and offsets the target grids by
`FRAME_RESCALE * window_start` — the same unit and rate the keyframe anchors
already use. `window_start` joins the layout signature, so two windows of the
same SHAPE at different clip positions cannot share a cached layout.

`window_start=0` reproduces the old behaviour exactly, which is what makes the
toolkit-side toggle a real A/B. `H3ContextWindows` has an
`absolute_window_positions` widget; `windowing.py` writes `window_start_frames`
into the payload per window.

Measured 2026-08-28, 192 frames, window 90 / overlap 39, three windows: before,
clean through window 0 then flickering; after, no flicker, motion smooth
throughout.

## The two settings that have to be right

Both found the hard way, and neither errors when wrong.

**`schedule` must be `standard_static`.** A uniform schedule re-derives every
window position on every step (`pad = round(num_frames * ordered_halving(step))`),
so no frame is rendered by a window in a consistent place — and with absolute
positions on, each frame is told a different time on every step.

**`causal_window_fix` must be OFF.** Core prepends an anchor frame to every
window after the first, making them 28 latent frames where the first is 27. 28
is not on H3's 5n+2 latent grid: 27 latent is exactly 90 pixel frames, 28 is 94,
which is not a legal run at all.

## Sizing

`H3ContextWindows` takes PIXEL frames and snaps both window and overlap to legal
runs (17n+5), because a clip's latent length is always 5k+2 and the final window
begins at `latent_t - window_len`. That lands on a multiple of 5 only when the
window length is itself a legal run.

`FRAME_PER_TOKEN` is `(1,4,4,4,4)` and `_video_t_spans` indexes it from k=0, so
PackedLayout assumes every window starts on a VAE chunk boundary. Both the
window length AND the stride must be multiples of 5 latent frames — 17 pixel
frames. The node enforces this and says so.

**Use overlap 39, not 22.** 90/22 gives a stride of 68 pixel frames, which is
off the audio grid (68 × 40/24 = 113.33 ticks). 90/39 gives 51, which divides
by 3 and lands on both clocks.

## Cost, measured against chunking

498 frames, 10 steps, 640x1120, window 57 latent (192 pixel) = the same size as
the chunks it was compared against:

    windowed    7:02    36.7 s/it   sampling 363s   overhead  59s   498 frames generated
    chunked     7:18    11.2 s/it   sampling 336s   overhead 102s   576 frames generated

Within 4% of each other. Chunking's sampling is faster per pass and it generates
16% more frames (the pin, three times) — about 0.58 s/frame against 0.74 — then
gives the lead back in fixed costs: three model restagings, three VAE decodes,
three audio decodes and the joins, against one of each.

**Speed is not the reason to choose between them.**

## Two null results, and what they rule out

Measured 2026-08-29, 192 frames, window 90 / overlap 39, three windows, same
seed and prompt, one variable at a time. Costs within noise of each other
(25.9, 26.0, 25.2 s/it), so neither is expensive enough to matter either way.

    pyramid        + freenoise      no visible difference
    overlap-linear + no freenoise   no visible difference

`freenoise` shuffles the noise so overlapping windows start from correlated
rather than independent noise. `overlap-linear` holds the fuse weight flat
across the window and ramps only inside the overlap, where `pyramid` shapes the
whole window as a triangle. Both are aimed squarely at seam quality. Neither
moved the background drift.

**So the drift is not a blending artifact.** The seam machinery combines two
windows faithfully; the windows GENERATED different backgrounds, and no
weighting fixes disagreement about content. That rules out every remaining
window setting — `closed_loop` is for looping and the audio fuse ramp is audio.

What is left attacks generation rather than combination: a reference image OF
THE SET (references are the mechanism that holds a thing steady, and in these
tests all three were the character, which is exactly why she was stable and the
room was not), and a room description with countable features so there is
something a window can get wrong. Untested.

## What windowing still cannot do

One conditioning for the whole clip. A sequence of prompts — four dialogue
clauses, a shot that changes what it is describing — collapses into a single
prompt applied to all of it. That is what sent this project to chunking.

Overlaps BLEND. Whatever crosses a seam is a weighted average of two windows'
output, not a cut.

## The open question: per-window conditioning

Core already has the mechanism, in `IndexListContextHandler.get_resized_cond`:

```python
if self.split_conds_to_windows and len(cond_in) > 1:
    region = window.get_region_index(len(cond_in))
    cond_in = [cond_in[region]]
```

Each window picks a conditioning by where its CENTRE falls in the clip:

```python
center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
region_idx   = int(center_ratio * num_regions)
```

Three prompts across three windows gives centres at roughly 0.19 / 0.50 / 0.81
→ regions 0, 1, 2.

`H3ContextWindows` currently passes `split_conds_to_windows=False` and does not
expose it. UNTESTED on H3, and two things are unknown:

- whether H3's payload survives the split. Each conditioning carries its own
  `minimax_payload` with its own `PackedLayout` built from its own `text_len`,
  and whether that gets rebuilt with the WINDOW's latent length rather than the
  full clip's is the same machinery the keyframe rebasing patches.
- how the region mapping behaves when windows outnumber prompts. Three prompts
  across nine windows puts the boundaries wherever the arithmetic lands, not
  where the clauses change.

And one thing that is known: the overlaps still blend, so a prompt change is a
39-frame crossfade rather than a cut. For an evolving description that is a
feature. For dialogue it is not — two prompts averaged across a seam do not give
clean speech out of the middle.

**Where this would win:** a shot whose DESCRIPTION evolves and where nobody
talks. One latent, no pin overhead, no hard seams, and a prompt that changes as
it goes. Worth testing there rather than on a dialogue clip.

## Reference blocks are CONTENT until the language model binds them

Measured 2026-08-29, and it settles a question worth having asked.

References reach the model twice. `tokenize` builds
`"<Picture 1>: <vision> <Picture 2>: <vision> ... <text>"`, so the LANGUAGE
MODEL sees the resized pixels as a prefix of vision tokens. Separately the DiT
receives the VAE latents, carried on the conditioning as `minimax_refs` and
packed as `ref_img` rows in its own sequence.

Per-window prompting needs one conditioning per prompt, and the expensive half
is Qwen3VL 32B re-processing that vision prefix every time. So `H3SwapPrompt`
encoded the new text ALONE and carried `minimax_refs` across — one full build
plus N-1 text-only encodes. It was 7-9x cheaper per extra prompt: 4.03s for the
full reference node against 0.60s and 0.44s for the swaps.

**It does not work, and the failure says why.** The output followed the prompt
briefly, then faded to reference image 1 and animated it, then to reference 2,
then to reference 3, and cut. The model rendered the references as SHOTS.

Three `ref_img` blocks sitting on the DiT's timeline with nothing in the text
binding them to a subject are just content to produce. The `<Picture N>` prefix
is what makes them identity, and `retention_analysis` naming `<Subject 1>` has
nothing to attach to when the language model never saw a picture.

It is the reference-video leak in its purest form: a reference the prompt does
not bind gets RENDERED rather than referenced.

**So every distinct prompt needs a full conditioning build.** At ~4s each on a
seven-minute render that is not the cost worth optimising — the cost that
mattered was `split_conds_to_windows` being off, which evaluated all three
conditionings in every window and took the sampler from 9 s/it to 100.

The node has been removed rather than left as a footgun.

## Related

- `patches/h3-window-absolute-positions.patch` — the core patch, apply/revert
  commands in `patches/README.md`
- `patches/h3-layout-cursor-logging.patch` — logs each layout's start `cond_t`,
  which is how the constant cursor was confirmed at runtime
- `windowing.py` — `H3EnableContextWindows`, `H3ContextWindows`, and the
  modality-dim hooks H3 needs because its audio latent puts time on dim 3
- `workflows/H3 09 - Windowed Long-form.json`
