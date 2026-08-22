# ComfyUI-H3-Toolkit

Nodes for MiniMax-H3: prompt construction, masked region replacement, subject
cropping, and long-form chaining.

The model already generates — ComfyUI's stock H3 nodes handle that. What this pack
adds is the knowledge around it: prompt format, the frame and audio grids, mask
geometry, and settings that were measured rather than guessed.

**Requires MiniMax-H3.** Several nodes reach into H3's paired (video, audio) latent
and its specific VAE geometry. They will not work with other video models.

## Install

Copy or symlink `h3_toolkit/` into `ComfyUI/custom_nodes/`, then restart.
Example workflows are in `workflows/` — copy them into
`ComfyUI/user/default/workflows/`.

Needs a ComfyUI recent enough to include **#15322**, which fixed H3's masked
sampling (shipped in v0.33.x). On anything older the masking nodes produce
swirling colour artifacts in the generated region, and no setting avoids it.

## Nodes

**`MiniMax H3/mask`** — `H3MatchSource`, `H3MaskInpaint`, `H3MaskStabilize`,
`H3SubjectCrop`, `H3SubjectUncrop`, `H3ApplyCrop`, `H3PreviewMaskCrop`,
`H3LatentPin`

Replace a masked region of an existing video while pinning everything outside it.

`H3MatchSource` conforms a clip to a canvas H3 can render — crop, fill, stretch or
pad, with a megapixel budget — and derives the width, height and legal frame count
for everything downstream. Its `mask_2` carries a second mask through the identical
conform, scale and **trim**, which rebuilding from stock resize nodes does not.

`H3MaskInpaint` builds the masked latent. `grow_px` dilates in pixel space before
the reduction, so the edge moves a pixel at a time rather than in the 16 px jumps
`dilate` is limited to. `forget_mask` controls how much of a region **forgets the
source it started from** — below denoise 1.0 the free region still starts holding
part of the source, and hue survives denoising better than structure does, so a
masked region can come back with the original's colour no matter what the prompt
says. Paint an area white there and it starts from noise while the rest of the mask
keeps the residual that holds pose.

`H3MaskStabilize` fills frame-to-frame dropouts. Segmentation loses whatever is
fast and small — hands, mostly — and each dropout flips a region between pinned and
free mid-clip. It uses morphology along the time axis rather than a median, because
a median asks "is this pixel usually masked", which for a moving subject is no: on a
hand crossing frame, a temporal median deletes it entirely.

`H3SubjectCrop` cuts the canvas down to the subject so the model renders fewer
tokens. How much that buys depends entirely on framing — a full-body subject on a
portrait canvas already fills the height, so the crop only gains horizontally. Its
info output reports the real saving. `H3PreviewMaskCrop` shows the mask **as the
model actually receives it**, which is coarser than the one you drew.

`H3SubjectCrop`'s `upscale_megapixels` turns the crop into a **detailer**: a face
in a wide frame occupies few latent cells, so cut it out, render the cut at the
budget the whole frame had, and paste it back. Same model, same cost, far more of
it spent on the face. `H3SubjectUncrop` reverses the scale. See workflow 08.

For a refinement pass, drop the **sigma shift**. Shift 12 is tuned for generating a
scene from nothing and compresses the schedule toward sigma 1, so even denoise 0.10
discards 40% of the source and no light pass is available. At shift 3 the same
denoise keeps 74%, and more of the steps land where detail is decided.

**`MiniMax H3/prompt`** — `H3ScenePrompt`, `H3LongFormLinks`, `H3PromptLint`,
`H3RewriterBrief`, `H3RewriterParse`

Build six-section prompts, and lint them against the traps below.

**`MiniMax H3/character`** — `H3Character`, `H3CharacterSave`

**`MiniMax H3/audio`** — `H3AudioPrompt`, `H3AudioLength`

**`MiniMax H3/video`** — `H3ReferenceToVideoLongForm`, `H3KeyframeTimeline`

Conditioning where a keyframe is also *presented* to the language model, and
keyframes that can coexist with reference images.

**`MiniMax H3/long-form`** — `H3Assemble`, `H3AudioSlice`, `H3AudioLock`,
`H3ChainFrame`, `H3Take`, `H3Resolution`

## Things that are easy to get wrong, and are enforced here

- Every link of a chain is generated INDEPENDENTLY. Relative wording ("continuing
  the same shot") has nothing to resolve against and re-invents the scene.
- Naming a failure tends to produce it. "no cuts, no camera movement" cut twice;
  the same setup without those words was clean at two seeds, one of which had cut.
- Retention binds to the SUBJECT, never to `<Picture N>`. Naming the picture as the
  retained thing makes the model render the anchor as a shot.
- With any reference present the prompt needs all six sections. The three-field form
  leaves the reference with no declared role and the model improvises.
- One soundtrack for a whole take, sliced per link. Generating audio per link
  restarts the music; pinning the same slice into every link loops it.
- A clip's length has to be legal on BOTH clocks. The video VAE takes 17n+5 frames;
  the audio latent runs at 40 Hz, and `audio_t = round(frames / 24 * 40)` only comes
  out exact when the frame count is also divisible by 3 — every third video run,
  51 apart: 39, 90, 141, 192, 243, 294, 345, 396. Off-grid lengths round, and the
  error accumulates across a chain.
- Masks reduce to latent space UNEVENLY in time. The VAE groups pixel frames
  `(1,4,4,4,4)`, so every fifth latent frame covers one pixel frame and the rest
  cover four. Splitting into equal buckets shifts every mask boundary by up to two
  frames and smears fast motion onto the tokens that should be sharpest.
- Below denoise 1.0 a masked region does not start blank. The sampler builds its
  starting point as `sigma*noise + (1-sigma)*source` over the whole frame, before
  any mask is applied, so the free region starts holding part of the source — 9% at
  denoise 0.45 under shift 12. That residual anchors position and structure, which
  is what keeps hands and contact points registered; it also carries appearance,
  which is what keeps the original's hair colour. Both move together on that one
  dial.
- Conform by CROPPING where you have the choice. In an inpaint most of the output
  *is* the source, so resampling softens pixels that were going to be kept verbatim.
  Where a downscale is unavoidable — 1080p into a 768p-class model — `fill` loses
  edges, `stretch` distorts, `pad` adds bars the model paints into.

## Layout

Modules are named for what they hold. Four carry no nodes at all:

| | |
|---|---|
| `timing.py` | the two clocks, the frame grids, and where they agree |
| `geometry.py` | crop rectangles |
| `cropplan.py` | where to cut a subject out of a clip |
| `avlatent.py` | reaching inside the paired (video, audio) latent |

The first three are torch-free so they can be tested without ComfyUI:

```
python3 test_geometry.py
python3 test_cropplan.py
python3 validate_workflows.py workflows/*.json
```

Run the validator after editing any workflow JSON by hand. Every workflow bug this
pack has shipped was structural and silent — ComfyUI drops a bad connection on load
rather than complaining, so the graph opens looking fine and fails at queue time.

## License

MIT — see `LICENSE`.
