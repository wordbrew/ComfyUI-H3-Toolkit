#!/usr/bin/env python3
"""Every custom type a node CONSUMES must be one something PRODUCES.

    H3ChunkRef.chunk_plan: received_type(H3_CHUNK_PLAN)
                           mismatch input_type(H3CHUNKPLAN)

I typed that type name from memory instead of reading it, and nothing caught
it until CJ hit a validation failure at submit — the worst place to find it,
because the graph composes fine and only core's type check refuses, after a
restart, with a message about two strings that look nearly identical.

A node pack's custom types are a tiny closed vocabulary. Any input asking for
a type no output in the pack emits is either a typo or a missing node, and
both are worth failing a suite over.
"""
import sys

sys.path.insert(0, ".")
import audit_nodes as A                                        # noqa: E402

classes, failed = A.load_pack()
assert not failed, failed

#: Core's own types. Anything else has to come from inside the pack.
BUILTIN = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "AUDIO", "VIDEO", "STRING", "INT", "FLOAT", "BOOLEAN", "SIGMAS",
    "SAMPLER", "GUIDER", "NOISE", "WEBCAM", "CONTROL_NET", "UPSCALE_MODEL",
    "DAMODEL", "*",
}

produced = set(BUILTIN)
for name, cls in classes.items():
    for t in getattr(cls, "RETURN_TYPES", ()) or ():
        if isinstance(t, str):
            produced.add(t)

fails = []
seen = 0
for name in sorted(classes):
    cls = classes[name]
    if not hasattr(cls, "INPUT_TYPES"):
        # a V3 io.ComfyNode declares define_schema() instead, and core
        # typechecks those itself from the schema objects -- there is no
        # string to get wrong by hand, which is the failure this guards
        continue
    try:
        spec = cls.INPUT_TYPES()
    except Exception as exc:                                    # noqa: BLE001
        fails.append(f"{name}.INPUT_TYPES() raised {exc!r}")
        continue
    for section in ("required", "optional"):
        for field, decl in (spec.get(section) or {}).items():
            t = decl[0] if isinstance(decl, (list, tuple)) and decl else decl
            if not isinstance(t, str):
                continue                      # a COMBO is a list of options
            seen += 1
            if t not in produced:
                near = sorted(x for x in produced
                              if x.replace("_", "") == t.replace("_", ""))
                fails.append(
                    f"{name}.{field} wants {t!r}, which no node emits"
                    + (f" — did you mean {near[0]!r}?" if near else ""))

print(f"checked {seen} typed input(s) across {len(classes)} node(s)")
print(f"the pack emits {len(produced) - len(BUILTIN)} custom type(s): "
      + ", ".join(sorted(produced - BUILTIN)))
print()
if fails:
    for f in fails:
        print("  FAIL " + f)
    print(f"\nFAIL — {len(fails)} input(s) ask for a type nothing produces")
    sys.exit(1)
print("types: every consumed type is one this pack produces")
