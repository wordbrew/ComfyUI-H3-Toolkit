"""A take as one HTTP request: plan it, or render it.

WHAT THIS IS FOR
  A front end should not have to know that a long-form H3 take is 27 wired
  nodes. It should send a script and a chunk count, learn exactly what it will
  get, and then ask for it. That is two calls, and this module is the half that
  does not involve aiohttp.

WHY A COMMITTED TEMPLATE AND NOT AN EXPORTED ONE
  An API-format graph addresses nodes by ID. Those ids belong to whoever
  exported the graph, and they move the moment that person edits it -- so a
  front end wired to node "156" breaks silently the next time the author
  rearranges anything. The template lives in templates/ under version control,
  and every node a caller can reach carries a ROLE in `_meta.title`
  (`h3.unet`, `h3.links`, `h3.ref.0` ...). Injection is keyed on the role, so
  ids are free to change and a missing role is an error naming the role rather
  than a KeyError on an integer.

  The template's wiring is CJ's working graph, which rendered 345 frames at
  14.375s headlessly before this module existed. Its model names are
  placeholders -- `REPLACE_ME_*` -- because the repo is public, so a render
  request MUST supply them. A placeholder reaching ComfyUI fails validation
  rather than rendering something wrong, which is the right way round.
"""
import copy
import json
import pathlib
import re

from .chunkplan import CONTINUITY, CUT_LIKE

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "templates"
DEFAULT_TEMPLATE = "longform_ref2va"
PLACEHOLDER = "REPLACE_ME"


#: The depth model every composition/motion reference shares. READ OFF CJ'S OWN
#: GRAPHS -- H3 10, 12 and 13 all load vitl_fp32 at auto precision. I had
#: guessed fp16 from "that is the sensible middle", which is exactly the kind
#: of invention that put three wrong aspect-ratio strings in the UI.
DEPTH_MODEL = "depth_anything_v2_vitl_fp32.safetensors"
DEPTH_PRECISION = "auto"


class TakeError(ValueError):
    """A request that cannot be turned into a graph, with the reason."""


def load_template(name=DEFAULT_TEMPLATE):
    p = TEMPLATE_DIR / f"{name}.json"
    if not p.is_file():
        have = sorted(x.stem for x in TEMPLATE_DIR.glob("*.json"))
        raise TakeError(f"no template {name!r}; have {have}")
    return json.loads(p.read_text(encoding="utf-8"))


def roles_of(graph):
    """role -> [node id]. A role may repeat; the caller decides if that is ok."""
    out = {}
    for nid, node in graph.items():
        role = (node.get("_meta") or {}).get("title") or ""
        if role.startswith("h3."):
            out.setdefault(role, []).append(nid)
    return out


def _one(graph, role, required=True):
    ids = roles_of(graph).get(role) or []
    if not ids:
        if required:
            raise TakeError(f"the template has no node for role {role!r}")
        return None
    if len(ids) > 1:
        raise TakeError(f"role {role!r} is on {len(ids)} nodes ({ids}); a role "
                        f"must address exactly one")
    return ids[0]


def _set(graph, role, field, value, required=True):
    nid = _one(graph, role, required)
    if nid is None:
        return False
    ins = graph[nid].setdefault("inputs", {})
    if isinstance(ins.get(field), list):
        raise TakeError(f"{role}.{field} is WIRED in the template, so a request "
                        f"cannot set it — unwire it or drop the field")
    ins[field] = value
    return True


# ------------------------------------------------------------------ planning
def plan_fields(req, planner):   # noqa: C901
    """-> (plan dict, lint list). `planner` is chunkplan.describe_count_plan et al.

    Kept separate from graph building so a front end can call /take/plan on
    every keystroke without constructing 27 nodes.
    """
    cf = int(req.get("chunk_frames", 192))
    cc = int(req.get("chunk_count", 3))
    ctx = int(req.get("context", 39))
    total, first, rest, notes = planner(cf, cc, ctx)
    beats = [b for b in str(req.get("beats", "")).splitlines() if b.strip()]
    lint = list(notes)
    if beats and len(beats) != cc:
        lint.append(f"{len(beats)} beat(s) for {cc} chunk(s) — one line per "
                    f"chunk, and chunk {cc} is the last one rendered")
    # the transition INTO each chunk; entry 0 of `continuity` is chunk 2's
    conts = req.get("continuity") or []
    if isinstance(conts, str):
        conts = [c.strip() for c in conts.replace(",", "\n").splitlines()
                 if c.strip()]
    chunks = []
    at = 0
    for i in range(cc):
        delivered = first if i == 0 else rest
        name = None if i == 0 else (conts[i - 1] if i - 1 < len(conts)
                                    else "carry")
        spec = CONTINUITY.get(name) if name else None
        chunks.append({
            "index": i,
            "number": i + 1,
            "frames": delivered,
            "start_seconds": round(at / 24.0, 3),
            "end_seconds": round((at + delivered) / 24.0, 3),
            "beat": beats[i] if i < len(beats) else None,
            "continuity": name,
            "shot": None,                    # filled below, once cuts are known
            "opens_shot": None,
            "continuity_what": spec["what"] if spec else None,
            "continuity_wired": bool(spec["ok"]) if spec else True,
            "carry_audio": (spec["audio"] == "carry") if spec else False,
        })
        at += delivered
    # SHOTS, the same rule scene_plan uses: a run from one cut to the next, and
    # head/tail belong to the shot rather than the chunk.
    shot = 0
    for i, c in enumerate(chunks):
        opens = i == 0 or (c["continuity"] in CUT_LIKE)
        if opens and i:
            shot += 1
        c["shot"], c["opens_shot"] = shot, opens
    for i, c in enumerate(chunks):
        if c["continuity"] and not c["continuity_wired"]:
            lint.append(f"chunk {c['number']}: {c['continuity']!r} is named but "
                        f"NOT WIRED — {CONTINUITY[c['continuity']].get('why','')}")
    return {
        "chunk_frames": cf, "chunk_count": cc, "context": ctx,
        "total_frames": total, "total_seconds": round(total / 24.0, 3),
        "first_chunk_frames": first, "other_chunk_frames": rest,
        "beats_required": cc, "chunks": chunks,
    }, lint


# ------------------------------------------------------------ graph building
def build(req, template=None):
    """A render request -> a submittable API graph. Raises TakeError with why."""
    graph = copy.deepcopy(template if template is not None
                          else load_template(req.get("template",
                                                     DEFAULT_TEMPLATE)))

    models = req.get("models") or {}
    for role, field, key in (("h3.unet", "unet_name", "unet"),
                             ("h3.clip", "clip_name", "clip"),
                             ("h3.vae.video", "vae_name", "vae"),
                             ("h3.vae.audio", "vae_name", "audio_vae")):
        if models.get(key):
            _set(graph, role, field, models[key])

    # STATIC LoRAs, AS MANY AS ASKED FOR. The template carries one loader and
    # this used to refuse a second, on the reasoning that more was "wiring the
    # template does not have". That was true of the export and false of the
    # model: a LoraLoaderModelOnly takes a MODEL and returns one, so they
    # chain, and minting the extra links is the same job set_references
    # already does for reference loaders.
    set_static_loras(graph, models.get("loras") or [])

    # REFERENCES ARE NOT CAPPED. `ref_images` is one autogrow input addressed as
    # ref_image_0, ref_image_1 ... with no declared maximum, so the number of
    # references is a property of the REQUEST, not of the template. This used to
    # raise above three, which was me over-applying a narrow finding: what was
    # measured is that a fourth reference FOR THE LOCATION wrecked subject
    # motion. Three CHARACTER references held. That is a note about what to put
    # in a reference, not a limit on how many exist, and it does not belong in
    # the code as an error.
    #
    # So: reuse the template's loaders for the first few, mint more when asked,
    # and PRUNE the leftovers -- an unused LoadImage still loads a file and
    # still sends it to the model, which is the one way this can silently send
    # a reference nobody asked for.
    set_references(graph, req.get("references") or [])
    # VOICES. The model's rule is that an audio reference accompanies a picture
    # or a video, so a voice with no pictures anywhere in the take is dropped
    # rather than sent on its own and silently ignored.
    voices = [v for v in (req.get("audio_references") or []) if v]
    if voices and not (req.get("references") or []):
        raise TakeError("a voice reference needs at least one picture in the "
                        "take: H3 pairs an audio reference with an image or a "
                        "video, and a lone voice is dropped")
    set_audio_references(graph, voices)

    # A CAST BEATS TWO TEXT BOXES. When `cast` is given it is compiled through
    # h3script and OVERRIDES subject_def_1 / retention_1, so a caller never
    # types `<Subject 2>` or a retention clause by hand. Explicit text still
    # wins if both are sent, because an escape hatch that cannot be reached is
    # not an escape hatch.
    if req.get("cast"):
        from . import h3script as _hs
        compiled = compile_cast(req["cast"], _hs.parse, _hs.emit)
        req = dict(req)
        req.setdefault("subject_def_1", compiled["subject_defs"])
        req.setdefault("retention_1", compiled["retention"])

    # the prompt, as head/beats/tail -- the fields H3LongFormLinks repeats into
    # every chunk, which is what makes each link independent
    # @HANDLES RESOLVE HERE, against the numbering the prompt will actually
    # use -- which is why it happens after the cast is compiled and from the
    # same sorted order. An author writes `@lily turns toward the window` and
    # never has to know she is Subject 1, or notice when adding a prop would
    # have made her Subject 2.
    hmap = handle_map(req.get("cast") or [])
    unknown = []
    for field in ("head", "beats", "tail", "subject_def_1", "retention_1",
                  "soundscape", "music", "task_type", "heads", "tails"):
        if field not in req:
            continue
        value = req[field]
        if hmap and isinstance(value, str):
            value, bad = apply_handles(value, hmap)
            unknown += bad
        _set(graph, "h3.links", field, value)

    if unknown:
        known = ", ".join("@" + k for k in sorted(hmap)) or "none"
        raise TakeError(
            f"{', '.join('@' + u for u in sorted(set(unknown)))} is not in the "
            f"cast. A handle that names nobody would reach the model as "
            f"literal text, so this stops rather than renders it. "
            f"Cast handles: {known}.")

    # PER-CHUNK REFERENCES COME LAST, because they WIRE `retention_1` and the
    # loop above SETS it. Doing them first meant a request carrying both a cast
    # and one reference raised "h3.links.retention_1 is WIRED in the template,
    # so a request cannot set it" -- _set correctly refusing to overwrite
    # wiring that this same function had just put there a dozen lines earlier.
    #
    # And the cast's retention text is not discarded to make room: it SEEDS the
    # chain as the first gate's `retention_in`, so the subjects' retention and
    # the references' clauses both reach the prompt. Dropping one for the other
    # would have been the quiet version of this bug.
    gate, ref_lint = set_chunk_references(
        graph, req.get("chunk_references") or [],
        seed=req.get("retention_1") or "") or (None, [])
    if gate is not None:
        links = _one(graph, "h3.links", required=False)
        if links is not None:
            # a LINK, not a value, so it bypasses _set deliberately
            graph[links].setdefault("inputs", {})["retention_1"] = [gate, 2]

    for field, key in (("chunk_frames", "chunk_frames"),
                       ("chunk_count", "chunk_count")):
        if key in req:
            _set(graph, "h3.plan", field, int(req[key]))

    # CONTINUITY HAS TO REACH THE PLAN, and it did not. Every transition set in
    # the editor drove the readout and nothing else: the graph got a uniform
    # plan with no transitions, so every chunk fell back to `carry` whatever the
    # timeline showed. The scenes a caller sends are the authority on this --
    # one entry per JOIN, so scene 2's transition is the first.
    conts = req.get("continuity")
    if not conts and req.get("scenes"):
        conts = [sc.get("continuity") or "carry"
                 for sc in req["scenes"][1:]]
    if conts:
        if isinstance(conts, (list, tuple)):
            conts = "\n".join(str(c or "carry") for c in conts)
        _set(graph, "h3.plan", "continuity", str(conts))

    # PER-SCENE LENGTHS, which is what makes the editor's drag-to-resize real.
    # Without this the graph took one chunk_frames for the whole take and every
    # scene rendered the same size however the timeline was drawn.
    if req.get("scenes"):
        sizes = [int(sc["frames"]) for sc in req["scenes"]
                 if str(sc.get("frames") or "").strip().isdigit()
                 or isinstance(sc.get("frames"), int)]
        if len(sizes) == len(req["scenes"]) and len(set(sizes)) > 1:
            _set(graph, "h3.plan", "scene_frames",
                 "\n".join(str(x) for x in sizes))
    if "context" in req:
        # a COMBO on the plan, an INT on Open; both have to agree or the carry
        # the plan describes is not the carry Open hands out
        _set(graph, "h3.plan", "context", str(int(req["context"])))
        _set(graph, "h3.chunk.open", "context_frames", int(req["context"]),
             required=False)

    if "lora_schedule" in req:
        _set(graph, "h3.lora.schedule", "schedule", req["lora_schedule"])
    if "seed" in req:
        _set(graph, "h3.noise", "noise_seed", int(req["seed"]))
    if "filename_prefix" in req:
        _set(graph, "h3.save", "filename_prefix", str(req["filename_prefix"]))
    if "megapixels" in req:
        _set(graph, "h3.resolution", "megapixels", float(req["megapixels"]),
             required=False)
    if "aspect_ratio" in req:
        _set(graph, "h3.resolution", "aspect_ratio", req["aspect_ratio"],
             required=False)

    left = placeholders_in(graph)
    if left:
        raise TakeError(
            "the template still holds placeholder(s) a request must replace: "
            + "; ".join(f"{r} = {v}" for r, v in left)
            + ". The repo is public, so model names are not committed.")
    return graph


def placeholders_in(graph):
    """-> [(role, value)] for every REPLACE_ME left. Empty means submittable."""
    out = []
    for nid, node in graph.items():
        role = (node.get("_meta") or {}).get("title") or nid
        for k, v in (node.get("inputs") or {}).items():
            if isinstance(v, str) and PLACEHOLDER in v:
                out.append((f"{role}.{k}", v))
    return out

# ------------------------------------------------------------------- the cast
#: Subjects are numbered in THIS order, always: people, then the place, then
#: things. CJ: "is there a way to always stack cast before setting and setting
#: before props? so we can always assume Subject 1 or Subject 2 are characters?"
#:
#: It is safe because subject numbering and PICTURE numbering both come from
#: one pass over the cast -- sort that pass and the two stay in step. Sorting
#: for display only is what would break it, and is the failure this pack keeps
#: paying for.
KIND_ORDER = {"person": 0, "library": 0, "place": 1, "setting": 1, "thing": 2}


def cast_kind(row):
    """A row's sort class, from whichever field it actually carries."""
    if row.get("setting") or row.get("kind") == "setting":
        return "place"
    sub = str(row.get("subkind") or "").lower()
    if sub in ("place", "setting"):
        return "place"
    if sub == "thing":
        return "thing"
    return "person"


def cast_order(cast):
    """People, then the place, then things. STABLE within each group.

    Stability matters: two characters keep their relative order, so adding a
    prop never renumbers them. A plain sort by kind with no tiebreak would be
    stable in CPython anyway, but saying so here is cheaper than finding out
    the day someone ports it.
    """
    return sorted(cast or [], key=lambda r: KIND_ORDER.get(cast_kind(r), 0))


def handle_map(cast):
    """{handle: subject number}, in the SAME order the prompt is numbered in.

    The handle is the row's `key` -- the thing the author types as `@lily`.
    Numbers are 1-based and match what h3script emits, because both walk the
    sorted list.
    """
    out = {}
    for i, row in enumerate(cast_order(cast)):
        key = str(row.get("key") or "").strip().lstrip("@").lower()
        if key:
            out[key] = i + 1
    return out


#: `@name` as an author writes it. Letters, digits, underscore and hyphen;
#: stops at anything else so `@lily,` and `@lily.` both resolve.
HANDLE_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_-]*)")


def apply_handles(text, mapping):
    """`@lily` -> `<Subject 1>`. -> (text, [unknown handles]).

    WHY THIS EXISTS
      So an author never types `<Subject 2>`. The number depends on how many
      subjects there are and in what order, which changes when a prop is added
      -- and a hand-typed number that is now wrong renders SILENTLY wrong: the
      model is told about a subject that is someone else. A handle cannot go
      stale, and one that names nobody is an error this can see.
    """
    unknown = []

    def sub(m):
        k = m.group(1).lower()
        if k in mapping:
            return f"<Subject {mapping[k]}>"
        unknown.append(m.group(1))
        return m.group(0)

    return HANDLE_RE.sub(sub, str(text or "")), unknown


def cast_to_script(cast):
    """Structured cast rows -> H3Script text. The ONE place that mapping lives.

    WHY GO THROUGH THE SCRIPT LANGUAGE AT ALL
      Because the numbering, the retention wording per KIND, and the store
      lookup are already written, tested and used by the ComfyUI panel. A UI
      that composed `<Subject 2> is ...` itself would be a second implementation
      of the pack's most consequential text, and the two would drift. So the
      rows become `@name = ...` lines and h3script.emit does the rest -- it is
      what turns `character Lily` into "<Subject 1> is Lily." with her anchors
      counted from the store, and what knows a SETTING retains "layout,
      architecture, materials and quality of light" while a person retains
      "facial identity, hair, eye colour and build".

    A row is {key, from_library | describe | setting, retention?, wears?,
              preserve?, allow?, pictures?, audio?}.
    """
    lines = []
    for i, row in enumerate(cast_order(cast)):
        key = str(row.get("key") or f"s{i + 1}").strip().lstrip("@") or f"s{i + 1}"
        if row.get("setting"):
            lines.append(f"@{key} = setting. {row['setting']}")
        elif row.get("from_library"):
            lines.append(f"@{key} = character {row['from_library']}")
        elif row.get("describe"):
            lines.append(f"@{key} = {row['describe']}")
        else:
            continue                      # an empty row is not a subject
        for field in ("retention", "wears", "preserve", "allow", "pictures",
                      "audio", "retention_detail"):
            if row.get(field) not in (None, "", []):
                lines.append(f"@{key}.{field} = {row[field]}")
    if not lines:
        return ""
    # emit() describes a take, so it needs at least one shot to describe. The
    # caller only wants the cast fields, and discards everything else.
    lines += ["", "shot | placeholder"]
    return "\n".join(lines)


def compile_cast(cast, parse, emit, lint=None):
    """-> {subject_defs, retention, counts, lint}. Empty cast gives empty text."""
    text = cast_to_script(cast)
    if not text:
        return {"subject_defs": "", "retention": "", "counts": {}, "lint": []}
    doc = parse(text)
    out = emit(doc)
    notes = []
    if lint is not None:
        # the script lint knows about a cast with no pictures, a speaker with no
        # voice, and the other traps; drop anything about the placeholder shot
        notes = [m for m in lint(doc, out) if "placeholder" not in m.lower()]
    return {
        "subject_defs": out.get("subject_defs", ""),
        "retention": out.get("retention", ""),
        "counts": out.get("counts", {}),
        "index": out.get("index", {}),
        "lint": notes,
    }

def set_static_loras(graph, loras):
    """Chain N static LoRAs between the attention backend and the sampler.

    The template wires exactly one:

        Model Attention Backend -> h3.lora.static -> ModelSamplingMiniMaxH3

    which is a chain of length one, not a limit. Each extra loader takes the
    previous one's MODEL, and whatever consumed the first one's output now
    consumes the last one's -- found by LINK rather than by role, so this
    keeps working if the template puts something else after the LoRA.

    Zero LoRAs is a real answer too: the loaders come out and the consumer is
    wired straight to the source, rather than left pointing at a loader
    holding a REPLACE_ME placeholder that fails validation.
    """
    slots = sorted((r for r in roles_of(graph) if r == "h3.lora.static"
                    or r.startswith("h3.lora.static.")),
                   key=lambda r: (len(r), r))
    ids = [n for r in slots for n in roles_of(graph)[r]]
    if not ids:
        if loras:
            raise TakeError("the template has no h3.lora.static node, so "
                            "static LoRAs cannot be wired")
        return

    first = ids[0]
    source = (graph[first].get("inputs") or {}).get("model")

    # whoever reads the LAST loader today is who must read the new last one
    tail = ids[-1]
    consumers = [(nid, field)
                 for nid, node in graph.items()
                 for field, v in (node.get("inputs") or {}).items()
                 if isinstance(v, list) and v[0] == tail]

    for nid in ids[1:]:
        graph.pop(nid, None)

    if not loras:
        graph.pop(first, None)
        for nid, field in consumers:
            graph[nid]["inputs"][field] = source
        return

    next_id = max((int(k) for k in graph if k.isdigit()), default=0) + 1
    prev, last = source, first
    for i, spec in enumerate(loras):
        if i == 0:
            nid = first
        else:
            nid = str(next_id); next_id += 1
            graph[nid] = {"class_type": graph[first]["class_type"],
                          "inputs": dict(graph[first].get("inputs") or {}),
                          "_meta": {"title": f"h3.lora.static.{i}"}}
        ins = graph[nid].setdefault("inputs", {})
        ins["model"] = prev
        ins["lora_name"] = spec["name"] if isinstance(spec, dict) else spec
        ins["strength_model"] = float(spec.get("strength", 0.8)) \
            if isinstance(spec, dict) else 0.8
        prev, last = [nid, 0], nid

    for nid, field in consumers:
        graph[nid]["inputs"][field] = [last, 0]


def set_audio_references(graph, names):
    """Wire exactly len(names) VOICE references into ref_audios.* -> [node ids].

    The conditioning node has carried `ref_audios` all along -- an Autogrow
    input beside `ref_images` -- and the template wired only the pictures. So a
    subject with a recorded voice had that voice saved, listed and described,
    and never sent: the model heard nothing and invented a voice.

    Unlike the picture loaders there is nothing in the template to copy, so the
    LoadAudio nodes are minted outright. `audio` is a COMBO over ComfyUI/input,
    which is why the sample has to be staged there first -- same reason the
    anchors are.
    """
    cond = _one(graph, "h3.conditioning", required=False)
    if cond is None:
        if names:
            raise TakeError("the template has no h3.conditioning node, so "
                            "voice references cannot be wired")
        return []
    ins = graph[cond].setdefault("inputs", {})

    # rebuild from scratch, and PRUNE the loaders behind the old wiring -- an
    # orphaned LoadAudio still executes and still fails on a missing file
    for key in [k for k in ins if k.startswith("ref_audios.ref_audio_")]:
        old = ins.pop(key)
        if isinstance(old, list) and graph.get(old[0], {}).get(
                "_meta", {}).get("title", "").startswith("h3.refaudio."):
            graph.pop(old[0], None)

    next_id = max((int(k) for k in graph if k.isdigit()), default=0) + 1
    made = []
    for i, name in enumerate(names):
        nid = str(next_id); next_id += 1
        graph[nid] = {"class_type": "LoadAudio",
                      "inputs": {"audio": name},
                      "_meta": {"title": f"h3.refaudio.{i}"}}
        ins[f"ref_audios.ref_audio_{i}"] = [nid, 0]
        made.append(nid)
    return made


#: What H3 accepts. The conditioning node declares these as Autogrow maxima and
#: a request past them fails validation, so they are checked here where the
#: message can say which reference was dropped and why.
REF_CAPS = {"image": 9, "video": 3, "audio": 3}

# the role vocabulary lives with the node that emits the clause, so the two
# cannot drift
from .chunkref import ROLES_FOR, canonical as canonical_role
from .chunkref import depth_for_role as role_wants_depth


def set_chunk_references(graph, refs, seed=""):
    """Wire per-chunk references: loader -> [trim] -> [depth] -> gate -> slot.

    Each entry is
        {source, kind: image|video, role, chunks, depth, start, length, label}

    and becomes its own little chain in the graph. The pieces are all stock --
    LoadImage / LoadVideo + GetVideoComponents, ImageFromBatch for the trim,
    DepthAnything_V2 for the depth pass -- and the only node of ours is the
    H3ChunkRef gate, which holds the reference to its chunks and emits the
    retention clause.

    THE DEPTH MODEL IS SHARED. One loader for every reference that needs it:
    it is the same weights and minting one per reference would load them again
    for each.

    THE GATES CHAIN THEIR RETENTION, because H3LongFormLinks has a single
    `retention_1` and an input takes one wire. The last gate's text goes there.
    -> the node id to wire into retention_1, or None.
    """
    if not refs:
        return None
    cond = _one(graph, "h3.conditioning", required=False)
    if cond is None:
        raise TakeError("the template has no h3.conditioning node, so "
                        "per-chunk references cannot be wired")
    ins = graph[cond].setdefault("inputs", {})

    open_node = _one(graph, "h3.chunk.open", required=False)
    if open_node is None:
        raise TakeError("the template has no h3.chunk.open node, so a "
                        "reference cannot know which chunk is rendering")
    plan_node = _one(graph, "h3.plan", required=False)

    # H3ChunkOpen's chunk_index and the plan, by NAME rather than by position --
    # the slot contract says outputs only ever get appended, so a name resolved
    # once here cannot drift the way a hardcoded index would.
    try:
        from .chunkrun import H3ChunkOpen
        outs = list(H3ChunkOpen.RETURN_NAMES)
        idx_slot = outs.index("chunk_index")
    except Exception:                                           # noqa: BLE001
        idx_slot = 4

    next_id = [max((int(k) for k in graph if k.isdigit()), default=0) + 1]

    def mint(cls, inputs, title=None):
        nid = str(next_id[0]); next_id[0] += 1
        graph[nid] = {"class_type": cls, "inputs": dict(inputs)}
        if title:
            graph[nid]["_meta"] = {"title": title}
        return nid

    depth_model = [None]

    def depth_for(src):
        if depth_model[0] is None:
            depth_model[0] = mint("DownloadAndLoadDepthAnythingV2Model",
                                  {"model": DEPTH_MODEL,
                                   "precision": DEPTH_PRECISION},
                                  "h3.depth.model")
        return mint("DepthAnything_V2",
                    {"da_model": [depth_model[0], 0], "images": src})

    counts = {"image": 0, "video": 0}
    last_ret, lint = None, []

    for i, r in enumerate(refs):
        kind = "video" if str(r.get("kind", "image")).lower() == "video" \
            else "image"
        src_name = str(r.get("source") or "").strip()
        if not src_name:
            lint.append(f"reference {i + 1} has no source; skipped")
            continue
        if counts[kind] >= REF_CAPS[kind]:
            lint.append(f"{src_name}: H3 takes at most {REF_CAPS[kind]} "
                        f"{kind} reference(s) — this one is DROPPED")
            continue

        if kind == "video":
            # THE PACK'S OWN LOADER, not LoadVideo + GetVideoComponents +
            # ImageFromBatch. VideoTrimLoad loads, trims and resizes in one
            # node, and its `spans` widget is the format the trim UI already
            # reads and writes -- `start count [label]`, one per line, which
            # lands in the workflow and therefore in the rendered mp4's
            # metadata as the only record of which part of which file a clip
            # came from. Three stock nodes could not say that.
            spans = str(r.get("spans") or "").strip()
            if not spans:
                st, ln = int(r.get("start") or 0), int(r.get("length") or 0)
                spans = f"{st} {ln}" if ln else "0 0"     # 0 0 = all of it
            out = [mint("VideoTrimLoad",
                        {"file": src_name, "spans": spans,
                         "span_index": int(r.get("span_index") or 0),
                         "force_rate": 24.0, "select_every_nth": 1},
                        f"h3.clip.{i}"), 0]
        else:
            out = [mint("LoadImage", {"image": src_name}), 0]

        # DEPTH FOLLOWS THE ROLE. It used to be a flag a caller could set
        # either way, which let a `composition` reference skip the pass and
        # carry the appearance the clause tells the model to ignore -- a
        # contradiction nobody should be able to express. `frame 0` is the one
        # role that must NOT be depth-filtered: it has to be the real picture
        # or it is not the frame.
        role = canonical_role(str(r.get("role") or "composition"))
        if role not in ROLES_FOR.get(kind, ()):
            allowed = ", ".join(ROLES_FOR.get(kind, ())) or "none"
            lint.append(f"{src_name}: a {kind} cannot be {role!r} — "
                        f"using {allowed.split(',')[0]}")
            role = (ROLES_FOR.get(kind) or ("composition",))[0]
        if role_wants_depth(role):
            out = [depth_for(out), 0]

        counts[kind] += 1
        gate_in = {"role": role,
                   "chunks": str(r.get("chunks") or "all"),
                   "chunk_index": [open_node, idx_slot],
                   "image": out,
                   "ordinal": counts[kind],
                   "label": str(r.get("label") or src_name)}
        if plan_node is not None:
            gate_in["chunk_plan"] = [plan_node, 0]
        if last_ret is not None:
            gate_in["retention_in"] = [last_ret, 2]
        elif seed:
            # the cast's own retention_analysis opens the chain, so the
            # subjects' wording survives alongside the references' clauses
            gate_in["retention_in"] = str(seed)
        gate = mint("H3ChunkRef", gate_in, f"h3.chunkref.{i}")
        last_ret = gate

        slot = "ref_videos.ref_video_" if kind == "video" \
            else "ref_images.ref_image_"
        ins[f"{slot}{counts[kind] - 1 if kind == 'video' else _next_image_slot(ins)}"] = [gate, 0]

    return last_ret, lint


def _next_image_slot(ins):
    """The first free ref_images index, so a gate lands after the cast's."""
    used = {int(k.rsplit("_", 1)[1]) for k in ins
            if k.startswith("ref_images.ref_image_")}
    i = 0
    while i in used:
        i += 1
    return i


def set_references(graph, names):
    """Wire exactly len(names) reference loaders, adding or pruning as needed."""
    cond = _one(graph, "h3.conditioning", required=False)
    if cond is None:
        if names:
            raise TakeError("the template has no h3.conditioning node, so "
                            "references cannot be wired")
        return
    ins = graph[cond].setdefault("inputs", {})
    existing = sorted((r for r in roles_of(graph) if r.startswith("h3.ref.")),
                      key=lambda r: int(r.rsplit(".", 1)[1]))
    loaders = [_one(graph, r) for r in existing]
    template_loader = graph[loaders[0]] if loaders else None

    # drop every ref_image_* wiring; it is rebuilt below
    for key in [k for k in ins if k.startswith("ref_images.ref_image_")]:
        del ins[key]

    next_id = max((int(k) for k in graph if k.isdigit()), default=0) + 1
    for i, name in enumerate(names):
        if i < len(loaders):
            nid = loaders[i]
        else:
            if template_loader is None:
                raise TakeError("the template wires no reference loader to copy")
            nid = str(next_id); next_id += 1
            graph[nid] = {"class_type": template_loader["class_type"],
                          "inputs": dict(template_loader.get("inputs") or {}),
                          "_meta": {"title": f"h3.ref.{i}"}}
        graph[nid].setdefault("inputs", {})["image"] = name
        graph[nid]["_meta"] = {"title": f"h3.ref.{i}"}
        ins[f"ref_images.ref_image_{i}"] = [nid, 0]

    # PRUNE. A loader left in the graph with nothing citing it is not inert:
    # ComfyUI executes it, it loads whatever file it holds, and on the template
    # that is a REPLACE_ME placeholder that fails validation for no good reason.
    for nid in loaders[len(names):]:
        graph.pop(nid, None)

    # THE REFERENCE SAMPLER KEEPS A SLOT, after the request's own references.
    # Clearing every ref_image_* above orphaned it -- the node stayed in the
    # graph, executed, and fed nothing. It emits None except at a
    # `reference_sample` join, so holding a slot permanently costs nothing on
    # any other chunk.
    sampler = _one(graph, "h3.refsample", required=False)
    if sampler is not None:
        ins[f"ref_images.ref_image_{len(names)}"] = [sampler, 0]
