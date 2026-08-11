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
