"""Schema guard for the one V3 node in this pack.

`H3ReferenceToVideoLongForm` is defined with `io.Schema` rather than the V1
`INPUT_TYPES` dict, because Autogrow is a DynamicInput with no V1 equivalent.
That buys nine growable reference slots and costs a dependency on ComfyUI's
schema API, so this checks the parts that would silently break saved workflows:
the node id, the output types, and the autogrow template's prefix and ceiling.

SKIPS cleanly when ComfyUI is not importable, so a bare checkout still runs the
rest of the suite. It only really tests something inside a Comfy install — or
against a checkout on sys.path, which is how it was developed.
"""

import sys

try:
    from comfy_api.latest import io  # noqa: F401
    import video
except Exception as exc:
    print(f"video schema: SKIPPED ({type(exc).__name__}: {exc})")
    print("  needs ComfyUI importable; run this from inside a Comfy install")
    sys.exit(0)

fails = []
N = video.H3ReferenceToVideoLongForm
sch = N.GET_SCHEMA()

# The node id is the only thing a saved workflow stores for an internal node.
if sch.node_id != "H3ReferenceToVideoLongForm":
    fails.append(f"node id changed to {sch.node_id!r} — breaks every saved workflow")

# Output slots are positional in saved links; appending is safe, reordering is not.
if tuple(N.RETURN_TYPES) != ("CONDITIONING", "LATENT"):
    fails.append(f"output types changed: {tuple(N.RETURN_TYPES)}")

# The required block is positional too, and workflow 03 wires all of it.
req = list(N.INPUT_TYPES().get("required", {}).keys())
want = ["clip", "vae", "audio_vae", "prompt", "width", "height", "length",
        "ref_image_size"]
if req != want:
    fails.append(f"required inputs changed:\n    got  {req}\n    want {want}")

opt = list(N.INPUT_TYPES().get("optional", {}).keys())
for name in ("ref_images", "keyframe", "keyframe_time", "present_keyframe"):
    if name not in opt:
        fails.append(f"optional input {name!r} missing")

# Autogrow does NOT appear as expanded slots in static INPUT_TYPES — the slots
# are materialised per-prompt from the live inputs. The template is what the
# frontend reads, so that is what is worth pinning.
ag = [i for i in sch.inputs if getattr(i, "id", None) == "ref_images"]
if not ag:
    fails.append("ref_images autogrow input is not in the schema")
else:
    tpl = ag[0].template
    if tpl.prefix != "ref_image_":
        fails.append(f"prefix {tpl.prefix!r} — saved workflows wire 'ref_image_N'")
    if tpl.min != 0:
        fails.append(f"min is {tpl.min}, should be 0: every reference is optional")
    if tpl.max != video.MAX_REF_IMAGES or tpl.max != 9:
        fails.append(f"max is {tpl.max}, should be 9 to match stock's ceiling")
    if tpl.input.io_type != "IMAGE":
        fails.append(f"template input is {tpl.input.io_type}, expected IMAGE")

# Registered through NODE_CLASS_MAPPINGS, ComfyUI's loader never calls
# GET_SCHEMA, so these must resolve lazily or the node lands uncategorised.
if N.CATEGORY != "MiniMax H3/video":
    fails.append(f"CATEGORY did not resolve: {N.CATEGORY!r}")
if not N.DESCRIPTION:
    fails.append("DESCRIPTION did not resolve")

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("video schema: ok (9 autogrow ref slots, id and outputs unchanged)")
