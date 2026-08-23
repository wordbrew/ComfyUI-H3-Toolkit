"""Reference-budget arithmetic, no torch and no ComfyUI needed.

Checks the sizing prediction against values worked out by hand from
comfy_extras/nodes_minimax_h3.py, so a drift in either shows up here rather
than in a report that quietly lies.
"""

import sys

from budget import _FALLBACK, frame_rows, ref_image_encoded_size

C = _FALLBACK
fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")


# --- frame_rows: the DiT's 2x2 patch grid over the /16 latent -------------- #
check("rows 640x1120", frame_rows(640, 1120), 20 * 35)
check("rows 832x832", frame_rows(832, 832), 26 * 26)
check("rows 1024x1024", frame_rows(1024, 1024), 32 * 32)
check("rows 2048x2048", frame_rows(2048, 2048), 64 * 64)
# a 32-multiple canvas is exactly (px/32) per axis
check("rows 768x1344", frame_rows(768, 1344), 24 * 42)

# --- match: shrink to the generation's pixel AREA, aspect preserved -------- #
# 1024x1024 into 640x1120: scale = sqrt(716800/1048576) = 0.8268 -> 846.7 -> 832
check("match 1024 sq", ref_image_encoded_size(1024, 1024, 640, 1120, "match", C),
      (832, 832))
# match never scales UP: a small reference stays put
check("match tiny", ref_image_encoded_size(256, 256, 640, 1120, "match", C),
      (256, 256))
# a portrait reference keeps its aspect to within the 32-px grid. 1080x1920
# lands on 640x1120 (0.571 vs 0.563) — rounding each axis to 32 independently
# MUST move the ratio a little, so this is a tolerance, not an equality.
tw, th = ref_image_encoded_size(1080, 1920, 640, 1120, "match", C)
check("match portrait size", (tw, th), (640, 1120))
assert abs(tw / th - 1080 / 1920) < 0.02, "aspect drifted more than the grid explains"
# and the encoded area lands near the generation's, which is what match means
assert 0.85 < (tw * th) / (640 * 1120) < 1.15

# --- max: native up to a 2048 short edge ----------------------------------- #
check("max 1024 sq", ref_image_encoded_size(1024, 1024, 640, 1120, "max", C),
      (1024, 1024))
check("max 2048 sq", ref_image_encoded_size(2048, 2048, 640, 1120, "max", C),
      (2048, 2048))
# 3072 short edge -> capped at 2048
check("max 3072 sq", ref_image_encoded_size(3072, 3072, 640, 1120, "max", C),
      (2048, 2048))
# a 4096x2048 landscape has short edge 2048, so it is NOT scaled
check("max wide at cap", ref_image_encoded_size(4096, 2048, 640, 1120, "max", C),
      (4096, 2048))
# max ignores the generation size entirely
check("max ignores gen", ref_image_encoded_size(1024, 1024, 4096, 4096, "max", C),
      ref_image_encoded_size(1024, 1024, 320, 320, "max", C))

# --- the shares this whole node exists to report --------------------------- #
def lt(f):
    return 2 if f <= 5 else ((f - 5) // 17) * 5 + 2


def share(nrefs, native, mode, frames, gen=(640, 1120)):
    tw, th = ref_image_encoded_size(native, native, gen[0], gen[1], mode, C)
    r = nrefs * frame_rows(tw, th)
    v = lt(frames) * frame_rows(*gen)
    return round(100 * r / (r + v), 1)


# the two measured renders
check("39f  3x match", share(3, 1024, "match", 39), 19.4)
check("294f 3x match", share(3, 1024, "match", 294), 3.2)
# what `max` buys at length
check("294f 3x max@1536", share(3, 1536, "max", 294), 10.2)
check("294f 3x max@2048", share(3, 2048, "max", 294), 16.8)
# max at 2048 on a long clip lands near the short clip's match ratio
assert abs(share(3, 2048, "max", 294) - share(3, 1024, "match", 39)) < 3.0

# --- length is the dominant term, not reference count ---------------------- #
# tripling the references does NOT recover a 7x dilution
check("294f 9x match", share(9, 1024, "match", 294), 9.1)
assert share(9, 1024, "match", 294) < share(3, 1024, "match", 39)

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("budget: all checks pass")
