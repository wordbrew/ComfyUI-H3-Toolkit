# MiniMax-H3 ComfyUI node pack

Install: symlink or copy `h3_audio/` into `ComfyUI/custom_nodes/`.
Example workflows: `workflows/` — copy into `ComfyUI/user/default/workflows/`.

Everything here encodes findings from the engine work in `../docs/long-form-waves.md`
and `../docs/prompting-ref2va.md`. The nodes do not wrap the model — ComfyUI's stock
H3 nodes already generate. What they add is the knowledge around it: prompt format,
timing, masking shape rules, and the settings that were measured rather than guessed.

## Nodes

**prompt** — `H3ScenePrompt`, `H3LongFormLinks`, `H3PromptLint`,
`H3RewriterBrief`, `H3RewriterParse`

**audio** — `H3AudioPrompt` (song / speech / instrumental, multi-speaker),
`H3AudioLength`

**character** — `H3Character`, `H3CharacterSave`

**long-form** — `H3MatchSource`, `H3Assemble`, `H3AudioSlice`, `H3Take`,
`H3Resolution`

Supporting modules with no nodes of their own: `timing.py` (the two clocks and the
runs where they agree) and `geometry.py` (the crop rules). Both are torch-free so
they can be tested — `python3 test_geometry.py`. `python3 validate_workflows.py
workflows/*.json` checks the example graphs, which is worth doing after any hand
edit; every workflow bug this pack has shipped was a silent structural one.

**video** — `H3ReferenceToVideoLongForm`, `H3KeyframeTimeline`, `H3AudioLock`,
`H3ChainFrame`, `H3LatentPin`, `H3MaskInpaint`

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
- Masking pins the source's own pixels, so the generated latent must match the
  source clip's shape exactly. `H3MatchSource` derives it.
- A clip's length has to be legal on BOTH clocks. The video VAE takes 17n+5 frames;
  the audio latent runs at 40 Hz, and `audio_t = round(frames / 24 * 40)` only comes
  out exact when the frame count is also divisible by 3. That is every third video
  run — 39, 90, 141, 192, 243, 294, 345, 396, spaced 51 apart. Off-grid lengths
  round, and the error accumulates across a chain. Our long-form work used 362,
  which is a legal video run and is a third of an audio step out; 345 is the aligned
  run nearest it. (Rule from `seitanism/ComfyUI-H3-Motion-Context-MultiRef`.)
- Conform by CROPPING, never by stretching. In an inpaint most of the output *is*
  the source, so resampling softens pixels that were going to be kept verbatim, and
  a non-uniform resize additionally hands the model a distorted body to match.
  Crop to the target aspect first, and only rescale if the size still differs.
