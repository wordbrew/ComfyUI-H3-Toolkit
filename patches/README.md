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

**Not reported upstream yet.**
