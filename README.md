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

**`MiniMax H3/mask`** — `H3MatchSource`, `H3MaskInpaint`, `H3SubjectCrop`,
`H3SubjectUncrop`, `H3ApplyCrop`, `H3PreviewMaskCrop`, `H3LatentPin`

Replace a masked region of an existing video while pinning everything outside it.
`H3SubjectCrop` cuts the canvas down to the subject so the model renders fewer
tokens; `H3PreviewMaskCrop` shows the mask **as the model actually receives it**,
which is coarser than the one you drew.

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
