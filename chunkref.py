"""Per-chunk reference gating, with the retention clause that makes it work.

WHY THIS EXISTS
  A reference is wired once and rides every chunk. That is right for identity --
  the same face should be the same face throughout -- and wrong for everything
  else. A composition reference for the opening shot has no business in chunk
  four, and a motion reference for one move should stop when the move does.

  More importantly, the model does not know what a reference is FOR. CJ's method
  is a depth pass plus a retention clause that says what to take from it:

    <Picture 5> is inspiration for frame 0, follow only the composition,
    ignore the subject in the center

  Depth-filtering the pixels is half of it. Without the clause the model reads a
  depth map as a thing to render. The two travel together, so this node emits
  both: the image, gated to its chunks, and the sentence that explains it.

WHAT IT DOES NOT DO
  It does not run the depth pass -- `DepthAnything_V2` does, upstream of here,
  which keeps the choice of depth model out of this pack. It does not know what
  slot it lands in either; it hands back an image and the caller wires it to
  `ref_images` or `ref_videos`.

THE ADDRESSING IS H3ChunkLora's
  Deliberately. `2`, `2-4`, `2,5`, `2+`, `last` and `all` mean the same thing in
  both places, and the parser is imported rather than reimplemented, because two
  spellings of the same syntax is how a schedule and a reference come to
  disagree about which chunk is which.
"""
from .chunklora import chunk_indices, parse_chunks

#: What each role tells the model to take from the reference. The wording is
#: CJ's, from graphs that work -- the clause is the load-bearing part and is not
#: worth paraphrasing.
ROLES = {
    "identity": {
        "tag": "Picture",
        "clause": "",            # identity is cited in subject_definitions
        "depth": False,
        "for": (),               # NOT offered for an additional reference
        "what": "the picture as itself — identity, folded into the subject "
                "definition. This is what a cast member's anchors already do, "
                "so it is not offered as an additional reference: make the "
                "person a subject instead.",
    },
    "frame 0": {
        "tag": "Picture",
        "clause": "{tag} is the first frame.",
        "depth": False,          # the actual image, or it is not the frame
        "for": ("image",),
        "what": "the image itself opens the chunk. No depth pass — it has to "
                "be the real picture or it is not the frame.",
    },
    "composition": {
        "tag": "Picture",
        "clause": "{tag} is inspiration for frame 0, follow only the "
                  "composition, ignore the subject in the center.",
        "depth": True,           # always; that is what separates framing from
                                 # appearance, and the clause assumes it
        "for": ("image",),
        "what": "the framing, not the person. Always goes through depth — "
                "that is the whole mechanism, and the clause above is written "
                "as though it did.",
    },
    "composition and motion": {
        "tag": "Video",
        "clause": "{tag} is used for inspiration for the new video, follow "
                  "only the composition and motion.",
        "depth": True,
        "for": ("video",),
        "what": "the movement and framing of a clip, not its appearance. "
                "Always goes through depth.",
    },
}

#: The roles a source of each kind may take. CJ: "Drop down for role should be
#: content smart and list only per image vs video." A clip has exactly one
#: useful role here -- extending a clip is a different workflow, not a chunk
#: reference -- so the picker is a statement rather than a question.
ROLES_FOR = {
    "image": ("composition", "frame 0"),
    "video": ("composition and motion",),
}

#: Depth is NOT a user toggle any more. It is a property of the role: a
#: composition reference that skipped the depth pass would carry the appearance
#: the clause tells the model to ignore, which is a contradiction no one should
#: be able to express by unticking a box.
def depth_for_role(role):
    return bool((ROLES.get(role) or {}).get("depth"))


ROLE_ORDER = ("composition", "frame 0", "composition and motion",
              "identity")


#: `motion` was the old name for `composition and motion`. A saved graph or a
#: stored reference may still say it, and it should keep working.
ALIASES = {"motion": "composition and motion"}


def canonical(role):
    return ALIASES.get(str(role or "").strip(), str(role or "").strip())


def retention_clause(role, ordinal):
    role = canonical(role)
    """The sentence that tells the model what to take. -> str, possibly empty."""
    spec = ROLES.get(role)
    if not spec or not spec["clause"]:
        return ""
    return spec["clause"].format(tag=f"<{spec['tag']} {max(1, int(ordinal))}>")


def active_here(chunks, chunk_index, total):
    """Is this reference live on this chunk? -> (bool, note).

    `all` or empty means every chunk, which is the old behaviour and the right
    default -- a reference someone added without thinking about scope should
    behave the way references always have.
    """
    text = str(chunks or "").strip().lower()
    if not text or text in ("all", "every", "*"):
        return True, ""
    spec = parse_chunks(text)
    if spec is None:
        return True, (f"{chunks!r} reads as a TIME, and a reference is scoped "
                      f"by CHUNK — write 2, 2-4 or last. Treating it as all.")
    want, bad = chunk_indices(spec, total)
    note = ""
    if bad:
        # the same reasoning as H3ChunkLora: a stale row must not kill a take
        note = (f"{chunks!r} names chunk(s) {', '.join(bad)} and the plan has "
                f"{total} — those are ignored, the take still renders.")
    return int(chunk_index) in want, note


class H3ChunkRef:
    """Hold a reference to certain chunks, and say what it is for."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (ROLE_ORDER, {"default": "composition",
                         "tooltip": "What the model should TAKE from this "
                                    "reference.\n\n"
                                    + "\n".join(f"{k} — {v['what']}"
                                                for k, v in ROLES.items())}),
                "chunks": ("STRING", {"default": "all",
                           "tooltip": "Which chunks this reference is live "
                                      "on.\n\n"
                                      "`all` (the default) is every chunk, "
                                      "which is how a reference has always "
                                      "behaved. Otherwise the same addressing "
                                      "H3 Chunk LoRA takes: `1`, `2-4`, `2,5`, "
                                      "`3+`, `last`.\n\n"
                                      "CHUNK NUMBERS, not seconds — a colon "
                                      "means time there and is refused here, "
                                      "because a reference that half-covers a "
                                      "chunk is not a thing the model can do."}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999,
                                "tooltip": "Which chunk is rendering. Wire it "
                                           "from H3 Chunk Open."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The reference itself. For a "
                                    "`composition` or `motion` role this "
                                    "should ALREADY have been through "
                                    "DepthAnything_V2 — this node gates and "
                                    "describes, it does not filter."}),
                "ordinal": ("INT", {"default": 1, "min": 1, "max": 32,
                            "tooltip": "Which <Picture N> or <Video N> this is "
                                       "in the presentation order, so the "
                                       "retention clause names the right one. "
                                       "References are numbered per type, from "
                                       "1, in the order they are wired."}),
                "chunk_plan": ("H3_CHUNK_PLAN", {"tooltip": "The plan, so `last` "
                               "and out-of-range chunks can be resolved."}),
                "label": ("STRING", {"default": "",
                          "tooltip": "A name for the report only. Never "
                                     "reaches the prompt."}),
                "retention_in": ("STRING", {"default": "", "forceInput": True,
                                 "tooltip": "The retention text from the "
                                            "reference BEFORE this one.\n\n"
                                            "Several references have to reach "
                                            "one `retention_1` on H3 Long-Form "
                                            "Links, and an input takes one "
                                            "wire — so they chain: each node "
                                            "appends its own clause and passes "
                                            "the whole thing on. Wire the last "
                                            "one's `retention` to the links "
                                            "node. Order here is presentation "
                                            "order."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("image", "active", "retention", "info")
    FUNCTION = "go"
    CATEGORY = "H3/longform"
    DESCRIPTION = ("Gate a reference to certain chunks and emit the retention "
                   "clause that says what to take from it.")
    #: `image` is legitimately unread when the reference is not live here
    AUDIT_UNREAD_OK = ("image",)

    def go(self, role, chunks, chunk_index, image=None, ordinal=1,
           chunk_plan=None, label="", retention_in=""):
        role = canonical(role)
        plan = (chunk_plan or {}).get("chunks") or []
        total = len(plan) or 1
        live, note = active_here(chunks, chunk_index, total)

        clause = retention_clause(role, ordinal) if live else ""
        who = (label or "").strip() or ROLES[role]["tag"].lower() + f" {ordinal}"

        lines = [f"H3 CHUNK REF — {who}, chunk {int(chunk_index) + 1} of {total}"]
        lines.append(f"  role {role}: {ROLES[role]['what']}")
        lines.append(f"  scope {chunks or 'all'} — "
                     + ("LIVE here" if live else "not this chunk, emitting None"))
        if clause:
            lines.append(f"  retention: {clause}")
        if role in ("composition", "motion"):
            lines.append("  expects a DEPTH pass upstream; a raw clip brings "
                         "its appearance with it")
        if note:
            lines.append("  " + note)
        if live and image is None:
            lines.append("  WARNING: live here but nothing is wired to `image`")

        # SPANNING A CUT IS ALLOWED, and warned about. A composition reference
        # carried across a cut is usually a mistake -- the new shot has its own
        # framing -- but it is a legitimate choice for a match cut or a repeated
        # setup, so this says so and gets out of the way. CJ: "leave it to the
        # user, i would say you warn them, but don't prevent."
        if live and plan and role in ("composition", "frame 0"):
            spec = parse_chunks(str(chunks or "").strip().lower()) \
                if str(chunks or "").strip().lower() not in ("", "all", "every",
                                                             "*") else None
            span = (chunk_indices(spec, total)[0] if spec
                    else set(range(total)))
            cuts = [k + 1 for k in sorted(span)
                    if k and k < len(plan) and plan[k].get("cut")]
            if cuts:
                lines.append(f"  NOTE: this spans a cut at chunk "
                             f"{', '.join(str(c) for c in cuts)}. The shot "
                             f"after a cut has its own framing, so a "
                             f"composition reference carried across one is "
                             f"usually not what you want — unless it is a "
                             f"match cut or the same setup returning.")

        # the chain: each reference appends to the one before it
        joined = "\n".join(x for x in (str(retention_in or "").strip(),
                                       clause) if x)

        return {"ui": {"h3char": ["\n".join(lines)]},
                "result": (image if live else None, live, joined,
                           "\n".join(lines))}


NODE_CLASS_MAPPINGS = {"H3ChunkRef": H3ChunkRef}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkRef": "H3 Chunk Reference"}
