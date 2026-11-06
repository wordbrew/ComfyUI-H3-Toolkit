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

#: THE NUMBER IN A CITATION IS NOT THE SLOT NUMBER, AND IT IS NOT FIXED.
#:
#: Core builds the list it presents to the tokeniser by walking the reference
#: slots and SKIPPING the empty ones (`if img is None: continue`, three times
#: over, for images, videos and audio). So `<Picture 4>` means "the fourth
#: image actually sent", and a gated reference that is dark on this chunk
#: renumbers every reference after it -- on this chunk only.
#:
#: Which means a citation cannot be computed when the graph is built. It is a
#: function of what is live, and what is live is a function of the chunk. The
#: old code wrote the number at build time from a counter that did not even
#: know about the cast's own references, so a frame-0 image sitting in slot 4
#: was cited as `<Picture 1>` -- the first cast anchor -- and the model
#: obligingly opened the take on a reference photo and then morphed away from
#: it.
#:
#: So the gates carry a TALLY. Each one adds itself if it is live, and each
#: one reads its own number off the tally it was handed. It rides the same
#: chain the retention text already uses, in its own type so it cannot be
#: wired into a text input by accident.
#:
#: `audios` counts the soundtracks emitted so far, because core emits a
#: clip's `<Audio j>` label immediately BEFORE its `<Video k>` -- so a voice
#: wired into ref_audios is numbered after every live soundtrack, not before.
TALLY_TYPE = "H3_REF_TALLY"

#: H3's audio markers, from the guide, sourced from the ONE list that already
#: holds them. Distinct from the visual ones and easy to confuse: `reference`
#: means take the CHARACTER of the sound, `fully_copy` means reuse the
#: recording itself.
try:
    from .prompt_lint import AUDIO_MARKERS as _AM
    AUDIO_MARKERS = tuple(_AM)
except Exception:                                               # noqa: BLE001
    AUDIO_MARKERS = ("fully_copy", "partially_copy", "reference",
                     "weak_reference")

#: What each one tells the model, in the guide's marker-then-how shape. This
#: lived in takeapi, which is the wrong way round now that the GATE emits a
#: soundtrack's clause at execution: a clip's <Audio j> and <Video k> are both
#: per-chunk facts that only the gate knows.
AUDIO_CLAUSE = {
    "fully_copy": "reuse this recording in full.",
    "partially_copy": "reuse part of this recording.",
    "reference": "take only the timbre, tone and cadence; do not reuse the "
                 "recording itself.",
    "weak_reference": "loose inspiration for the sound only; diverge freely.",
}


def empty_tally(images=0, videos=0, audios=0):
    return {"images": int(images), "videos": int(videos),
            "audios": int(audios)}


def tally_of(value, images=0):
    """Whatever came down the wire -> a tally. Never raises.

    `images` seeds the picture count with the references that are wired
    OUTSIDE the chain and are always live -- the cast's own anchors. They
    occupy the first slots and are never dark, so they are a constant the head
    of the chain starts from.
    """
    if isinstance(value, dict):
        return empty_tally(value.get("images", 0), value.get("videos", 0),
                           value.get("audios", 0))
    return empty_tally(images)

#: What each role tells the model to take from the reference. The wording is
#: CJ's, from graphs that work -- the clause is the load-bearing part and is not
#: worth paraphrasing.
ROLES = {
    "identity": {
        "tag": "Picture",
        "clause": "",            # identity is cited in subject_definitions
        "depth": False,
        "for": ("image",),       # reachable, but see ROLES_OFFERED
        "what": "the picture as itself — identity, folded into the subject "
                "definition. This is what a cast member's anchors already do, "
                "so it is not offered as an additional reference: make the "
                "person a subject instead.",
    },
    "frame 0": {
        "tag": "Picture",
        # fully_preserved, not attribute_transfer: this image is not a
        # QUALITY being transferred, it is the thing itself, unchanged. And
        # "continues from it" because the recorded failure here is the
        # opposite -- the reference sitting as the first five frames and then
        # the real shot starting, which is a still followed by a cut rather
        # than an opening. See the parked frame-0 investigation.
        # "the video", not "the take" -- the model's own vocabulary, and the
        # same word CJ's proven motion clause already uses ("inspiration for
        # the new video"). `take` is this app's word for the thing being
        # made; it is not a word the guide puts in a prompt.
        "clause": "{tag}: fully_preserved - is the first frame of the video; "
                  "the opening frame matches it exactly and the shot "
                  "continues from it.",
        "depth": False,          # the actual image, or it is not the frame
        "for": ("image",),
        "what": "the image itself opens the chunk. No depth pass — it has to "
                "be the real picture or it is not the frame.",
    },
    "last frame": {
        "tag": "Picture",
        # The mirror of `frame 0`, and the same reasoning: the image is not a
        # quality being transferred, it is the thing itself. "arrives at it"
        # because the failure to avoid is the model treating a closing anchor
        # as a second shot tacked on the end -- the clause has to say the
        # motion LANDS there.
        "clause": "{tag}: fully_preserved - is the final frame of the video; "
                  "the shot moves toward it and the closing frame matches it "
                  "exactly.",
        "depth": False,          # the actual image, or it is not the frame
        "for": ("image",),
        "what": "the image itself closes the chunk. No depth pass — it has to "
                "be the real picture or it is not the frame. Pairs with "
                "`frame 0` to render a move between two stills.",
    },
    # A REFERENCE THAT IS JUST A REFERENCE. CJ: "Or just as a general
    # reference (no depth or frame call out)." Every other role here says
    # something specific about what to take, which is right when there IS
    # something specific -- and wrong when the picture is simply a thing the
    # shot should resemble. This one names the marker and stops.
    #
    # The marker is the user's, not the role's, because that is the one choice
    # a general reference actually has: the library already offers all four
    # for a subject, and a reference is the same decision about a picture that
    # is not a subject.
    "reference": {
        "tag": "Picture",        # overridden per kind; a clip gets <Video N>
        # `{how}` IS THE WHOLE SECOND HALF, and it is the user's to write.
        # CJ: "i dont necessarily want the additional reference to be an
        # identity reference, tied to a cast member. but perhaps I want it to
        # be anatomy or something like that, so how do we handle that?"
        #
        # retention_analysis's form is `marker - how`, so the answer is not a
        # role per attribute -- anatomy, wardrobe, hair, a prop's markings,
        # the quality of light -- it is one role whose `how` says which. The
        # marker sets how hard it holds; the sentence says what to take.
        "clause": "{tag}: {marker} - {how}",
        "depth": False,          # no depth, and no frame claimed
        "for": ("image", "video"),
        "what": "a picture or clip the shot should resemble. Say what to "
                "take from it in your own words — \"the body proportions and "
                "anatomy, not the face\" — and pick how hard it holds. "
                "Nothing is assumed about framing, motion or which frame it "
                "lands on.",
    },
    "composition": {
        "tag": "Picture",
        # THE MARKER LEADS. retention_analysis's form is `marker - how`, and
        # these went in as a bare sentence carrying the `how` alone. The
        # sentence is CJ's, from graphs that work, and is unchanged -- only
        # the marker is added, which is what makes the A/B a single variable.
        "clause": "{tag}: attribute_transfer - is inspiration for frame 0, "
                  "follow only the composition, ignore the subject in the "
                  "center.",
        "depth": True,           # always; that is what separates framing from
                                 # appearance, and the clause assumes it
        "for": ("image",),
        "what": "the framing, not the person. Always goes through depth — "
                "that is the whole mechanism, and the clause above is written "
                "as though it did.",
    },
    "composition and motion": {
        "tag": "Video",
        "clause": "{tag}: attribute_transfer - is used for inspiration for "
                  "the new video, follow only the composition and motion.",
        "depth": True,
        "for": ("video",),
        "what": "the movement and framing of a clip, not its appearance. "
                "Always goes through depth.",
    },
}

#: WHAT EACH MARKER ASKS FOR, in words. The marker alone is the model's
#: vocabulary; retention_analysis's form is `marker - how`, so a clause that
#: stopped at the marker would be half a line. These are the `how`.
MARKER_HOW = {
    "fully_preserved": "everything it shows, unchanged",
    "partially_preserved": "its main features, letting the shot vary the rest",
    "attribute_transfer": "its qualities only — colour, texture, light, "
                          "material — not the thing itself",
    "weak_reference": "loose inspiration",
}

#: The default for a general reference. The weakest of the four on purpose: a
#: picture someone added without saying what it was for should nudge the shot,
#: not govern it.
DEFAULT_MARKER = "weak_reference"

MARKERS = tuple(MARKER_HOW)


#: A clip cited as a general reference is `<Video N>`, not `<Picture N>`. Every
#: other role is single-kind so its tag is FIXED by the role, and `kind` must
#: not be able to override it -- a `composition and motion` reference is a
#: <Video N> whatever anyone passes.
TAG_FOR_KIND = {"image": "Picture", "video": "Video"}

#: The roles whose tag the kind decides. Exactly one, by construction: a role
#: that accepts both kinds is the only role that cannot name its own tag.
TAG_BY_KIND_ROLES = ("reference",)


def tag_for(role, kind=""):
    spec = ROLES.get(canonical(role)) or {}
    if canonical(role) in TAG_BY_KIND_ROLES:
        return TAG_FOR_KIND.get(str(kind or "").lower()) or spec.get("tag", "Picture")
    return spec.get("tag", "Picture")


#: The roles a source of each kind may take. CJ: "Drop down for role should be
#: content smart and list only per image vs video." A clip has exactly one
#: useful role here -- extending a clip is a different workflow, not a chunk
#: reference -- so the picker is a statement rather than a question.
ROLES_FOR = {
    "image": ("composition", "frame 0", "last frame", "reference", "identity"),
    "video": ("composition and motion", "reference"),
}

#: ...but it is still not OFFERED as an additional reference. A person who
#: should be recognised is a cast member; `identity` reaches this gate only
#: because a cast member scoped to part of the take is a gated reference.
ROLES_OFFERED = {
    "image": tuple(r for r in ROLES_FOR["image"] if r != "identity"),
    "video": ROLES_FOR["video"],
}

#: Depth is NOT a user toggle any more. It is a property of the role: a
#: composition reference that skipped the depth pass would carry the appearance
#: the clause tells the model to ignore, which is a contradiction no one should
#: be able to express by unticking a box.
def depth_for_role(role):
    return bool((ROLES.get(role) or {}).get("depth"))


ROLE_ORDER = ("composition", "frame 0", "last frame", "reference",
              "composition and motion", "identity")


#: `motion` was the old name for `composition and motion`. A saved graph or a
#: stored reference may still say it, and it should keep working.
ALIASES = {"motion": "composition and motion"}


def canonical(role):
    return ALIASES.get(str(role or "").strip(), str(role or "").strip())


def default_how(marker=""):
    """The sentence a general reference gets when nobody wrote one."""
    m = str(marker or "").strip()
    m = m if m in MARKER_HOW else DEFAULT_MARKER
    return f"is a general visual reference for the video; take " \
           f"{MARKER_HOW[m]} from it."


def retention_clause(role, ordinal, marker="", kind="", how=""):
    """The sentence that tells the model what to take. -> str, possibly empty.

    `marker`, `kind` and `how` matter only for the `reference` role: it is the
    one role whose marker the user chooses, the one that can cite either a
    picture or a clip, and the one whose sentence is written rather than
    fixed. Every other role settles all three, and passing them is harmless.
    """
    role = canonical(role)
    spec = ROLES.get(role)
    if not spec or not spec["clause"]:
        return ""
    tag = tag_for(role, kind)
    m = str(marker or "").strip() or DEFAULT_MARKER
    if m not in MARKER_HOW:
        m = DEFAULT_MARKER
    text = str(how or "").strip() or default_how(m)
    # A SENTENCE, ending in a full stop. The guide's clauses all do, and a
    # fragment run into the next clause is the kind of thing that reads as one
    # instruction about the wrong reference.
    if text and text[-1] not in ".!?":
        text += "."
    return spec["clause"].format(tag=f"<{tag} {max(1, int(ordinal))}>",
                                 marker=m, how=text)


def opens_shot(chunk, index=None):
    """Does this chunk START a shot? -> bool.

    `cut` is H3ChunkPlan's key and `opens_shot` is the take planner's; they
    mean the same thing and a plan may carry either. Reading only one made a
    plan from the other source look like one unbroken shot, which silently
    turned `frame 0`'s default scope into chunk 1 alone.
    """
    c = chunk or {}
    if index == 0:
        return True
    return bool(c.get("cut") or c.get("opens_shot"))


def _openers(plan):
    """Chunk indices whose opening frames are FRESH -- chunk 1 and every cut."""
    return [i for i, c in enumerate(plan or []) if opens_shot(c, i)]


def _closers(plan):
    """Chunk indices whose CLOSING frames are the end of a shot.

    The mirror of `_openers`: the final chunk, and any chunk the next one cuts
    away from. A `last frame` anchor belongs on one of these, for the same
    reason `frame 0` belongs on an opener -- it is where the shot actually
    ends rather than where it hands off.
    """
    n = len(plan or [])
    return [i for i in range(n)
            if i == n - 1 or opens_shot(plan[i + 1], i + 1)]


def default_scope(role, plan):
    """What a role should cover when nobody said. -> a chunks spec.

    `frame 0` defaults to the shot openers rather than `all`, because `all`
    would place it on continuing chunks where it is refused anyway -- a
    default that is wrong everywhere but the first chunk is a default that
    teaches the wrong thing.
    """
    if not plan or role not in ("frame 0", "last frame"):
        return "all"
    op = _openers(plan) if role == "frame 0" else _closers(plan)
    return ",".join(str(i + 1) for i in op) if op else "1"


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
                # APPENDED, per the slot contract.
                "marker": (MARKERS, {"default": DEFAULT_MARKER,
                           "tooltip": "How hard the model should hold to this "
                                      "reference. Only the `reference` role "
                                      "reads it — every other role fixes its "
                                      "own marker, because the marker is part "
                                      "of what the role MEANS.\n\n"
                                      + "\n".join(f"{k} — take {v}"
                                                  for k, v in MARKER_HOW.items())}),
                "kind": (("image", "video"), {"default": "image",
                         "tooltip": "Whether this reference is cited as "
                                    "<Picture N> or <Video N>. Only the "
                                    "`reference` role can be either; the rest "
                                    "are one or the other by definition."}),
                # APPENDED, per the slot contract.
                "tally_in": (TALLY_TYPE, {"tooltip":
                    "The running count of references already cited, from the "
                    "gate BEFORE this one.\n\n"
                    "WHY A COUNT HAS TO TRAVEL: core presents only the "
                    "references it actually received, skipping empty slots, "
                    "so `<Picture 4>` means the fourth picture SENT — and a "
                    "gated reference that is dark on this chunk renumbers "
                    "every one after it. The number therefore cannot be "
                    "written when the graph is built. Chain this alongside "
                    "`retention_in`, in slot order, and each gate reads its "
                    "own number off it."}),
                "cast_images": ("INT", {"default": 0, "min": 0, "max": 9,
                    "tooltip": "How many reference images are wired OUTSIDE "
                               "this chain and always live — the cast's own "
                               "anchors. They hold the first slots and are "
                               "never dark, so they are where the count "
                               "starts. Only the FIRST gate in the chain "
                               "needs it."}),
                "with_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "This clip's own soundtrack is sent too.\n\n"
                               "It takes an <Audio j> label of its own, and "
                               "core emits that label immediately BEFORE the "
                               "clip's <Video k> — which is why a voice wired "
                               "into ref_audios is numbered AFTER every live "
                               "soundtrack. Gating the clip off drops its "
                               "sound with it."}),
                "audio_marker": (("fully_copy", "partially_copy", "reference",
                                  "weak_reference"), {"default": "reference",
                    "tooltip": "How much of that soundtrack to take. "
                               "`reference` takes the timbre and cadence "
                               "without reusing the recording."}),
                "how": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "What to take from this reference, in your own "
                               "words — \"the body proportions and anatomy, "
                               "not the face\".\n\n"
                               "Only the `reference` role reads it; every "
                               "other role has its own sentence, which is "
                               "most of what the role MEANS. Left empty it "
                               "says it is a general reference and leans on "
                               "the marker alone."}),
            },
        }

    # `tally` is APPENDED, per the slot contract: a new output goes on the
    # END or every saved graph silently rewires.
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING", TALLY_TYPE)
    RETURN_NAMES = ("image", "active", "retention", "info", "tally")
    FUNCTION = "go"
    CATEGORY = "H3/longform"
    DESCRIPTION = ("Gate a reference to certain chunks and emit the retention "
                   "clause that says what to take from it.")
    #: `image` is legitimately unread when the reference is not live here
    AUDIT_UNREAD_OK = ("image",)

    def go(self, role, chunks, chunk_index, image=None, ordinal=1,
           chunk_plan=None, label="", retention_in="",
           marker=DEFAULT_MARKER, kind="image",
           tally_in=None, cast_images=0, with_audio=False,
           audio_marker="reference", how=""):
        role = canonical(role)
        plan = (chunk_plan or {}).get("chunks") or []
        total = len(plan) or 1
        live, note = active_here(chunks, chunk_index, total)
        who = (label or "").strip() \
            or tag_for(role, kind).lower() + f" {ordinal}"
        lines = [f"H3 CHUNK REF — {who}, chunk {int(chunk_index) + 1} of {total}"]

        # FRAME 0 BELONGS TO A CHUNK THAT OPENS A SHOT. Chunk 1 always does,
        # and so does any chunk after a cut -- its opening frames are fresh.
        # A CONTINUING chunk's opening frames ARE the pin: the previous
        # chunk's tail, reproduced and masked so the model reads those rows as
        # already finished. An image anchored there sets two things claiming
        # the same frames -- the mask saying "settled, do not denoise" and the
        # keyframe saying "look like this". The hard hold is load-bearing and
        # a chunk that drifts unheld makes the seam the clip's largest step,
        # so this is a conflict rather than a preference.
        #
        # Unlike spanning a cut, there is no reading where it is what you
        # meant, so it REFUSES rather than warns.
        if role == "frame 0" and plan and live:
            k = int(chunk_index)
            if not (k == 0 or (k < len(plan) and opens_shot(plan[k], k))):
                lines.append(
                    f"  REFUSED: chunk {k + 1} CONTINUES the shot before it, "
                    f"so its opening frames are the pin — the previous "
                    f"chunk's tail, held by the mask. An image anchored there "
                    f"competes with that hold. Scope this to a chunk that "
                    f"opens a shot: "
                    f"{', '.join(str(i + 1) for i in _openers(plan)) or '1'}.")
                live = False

        # A `last frame` ANCHOR MID-SHOT IS A DIRECTION, NOT A CONFLICT --
        # which is why this warns where `frame 0` refuses. The frame-0 clash is
        # with the PIN, and the pin lands on a continuing chunk's opening rows;
        # nothing is held at a chunk's tail. So "arrive at this pose by the end
        # of chunk 2, then carry on in chunk 3" is a thing someone can mean. It
        # is also easy to mean by accident, so it says so.
        if role == "last frame" and plan and live:
            cl = _closers(plan)
            if int(chunk_index) not in cl:
                lines.append(
                    f"  NOTE: chunk {int(chunk_index) + 1} is CONTINUED by the "
                    f"one after it, so this makes the shot land on the image "
                    f"mid-move and then carry on. That is a legitimate "
                    f"direction; if you meant the end of the shot, scope it to "
                    f"{', '.join(str(i + 1) for i in cl) or '1'}.")

        # ---- THE NUMBER. Read off the tally rather than written at build
        # time, because core presents only the references it actually got and
        # a dark gate renumbers everything after it. `ordinal` survives as the
        # fallback for a hand-built graph with nothing chained in.
        family = "video" if tag_for(role, kind) == "Video" else "image"
        tally = tally_of(tally_in, cast_images)
        # NOTHING CHAINED IN MEANS TRUST THE HUMAN. A graph built by hand sets
        # `ordinal` on the first gate and chains the rest, so an unseeded
        # tally starts from it rather than from 1 -- otherwise wiring the
        # chain up correctly would be the only way to use the node at all,
        # and every existing saved graph would start citing <Picture 1>.
        seeded = tally_in is not None or int(cast_images) > 0
        if not seeded:
            tally[family + "s"] = max(0, int(ordinal) - 1)
        audio_no = 0
        if live:
            if family == "video":
                # THE SOUNDTRACK'S LABEL COMES FIRST. Core emits a clip's
                # <Audio j> immediately before its <Video k>, so the audio
                # number is taken before the video number -- the same order,
                # or every voice downstream is off by one.
                if with_audio:
                    tally["audios"] += 1
                    audio_no = tally["audios"]
                tally["videos"] += 1
                my_no = tally["videos"]
            else:
                tally["images"] += 1
                my_no = tally["images"]
        else:
            my_no = int(ordinal)
        if not seeded:
            lines.append(f"  no tally chained in — numbering from `ordinal`, "
                         f"so <{tag_for(role, kind)} {my_no}> is only right "
                         f"if you counted the live references yourself")

        clause = retention_clause(role, my_no, marker, kind, how) if live else ""
        lines.append(f"  role {role}: {ROLES[role]['what']}")
        if live:
            lines.append(f"  cited as <{tag_for(role, kind)} {my_no}> "
                         f"(images {tally['images']}, videos "
                         f"{tally['videos']}, audio {tally['audios']} so far)")
        lines.append(f"  scope {chunks or 'all'} — "
                     + ("LIVE here" if live else "not this chunk, emitting None"))
        if clause:
            lines.append(f"  retention: {clause}")
        if role in ("composition", "motion"):
            lines.append("  expects a DEPTH pass upstream; a raw clip brings "
                         "its appearance with it")
        # ---- THE SOUNDTRACK'S OWN CLAUSE. It used to be written in the
        # builder, which could not know either number: it said
        # `<Video {slot + 1}>` from the SLOT, and counted `<Audio j>` once for
        # the whole take. Both are per-chunk facts, and both are known here.
        if live and family == "video" and with_audio:
            am = audio_marker if audio_marker in AUDIO_CLAUSE else "reference"
            clause = "\n".join(x for x in (
                clause,
                f"<Audio {audio_no}>: {am} - {AUDIO_CLAUSE[am]} It is the "
                f"soundtrack of <Video {my_no}>.") if x)
            lines.append(f"  soundtrack cited as <Audio {audio_no}>, "
                         f"marker {am}")
        elif with_audio and not live:
            # core reads a clip's soundtrack INSIDE the video loop, after it
            # skips a None video -- so gating the clip off drops its sound
            # silently. Saying so is the whole fix available here.
            lines.append("  its soundtrack goes with it: core only reads a "
                         "clip's audio alongside the clip, so nothing of this "
                         "sound reaches this chunk")

        if note:
            lines.append("  " + note)
        if live and image is None:
            lines.append("  WARNING: live here but nothing is wired to `image`")

        # SPANNING A CUT IS ALLOWED, and warned about. A composition reference
        # carried across a cut is usually a mistake -- the new shot has its own
        # framing -- but it is a legitimate choice for a match cut or a repeated
        # setup, so this says so and gets out of the way. CJ: "leave it to the
        # user, i would say you warn them, but don't prevent."
        if live and plan and role in ("composition", "frame 0", "last frame"):
            spec = parse_chunks(str(chunks or "").strip().lower()) \
                if str(chunks or "").strip().lower() not in ("", "all", "every",
                                                             "*") else None
            span = (chunk_indices(spec, total)[0] if spec
                    else set(range(total)))
            cuts = [k + 1 for k in sorted(span)
                    if k and k < len(plan) and opens_shot(plan[k], k)]
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
                           "\n".join(lines), tally)}


NODE_CLASS_MAPPINGS = {"H3ChunkRef": H3ChunkRef}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkRef": "H3 Chunk Reference"}
