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

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "templates"
DEFAULT_TEMPLATE = "longform_ref2va"
PLACEHOLDER = "REPLACE_ME"


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
def plan_fields(req, planner):
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
    chunks = []
    at = 0
    for i in range(cc):
        delivered = first if i == 0 else rest
        chunks.append({
            "index": i,
            "number": i + 1,
            "frames": delivered,
            "start_seconds": round(at / 24.0, 3),
            "end_seconds": round((at + delivered) / 24.0, 3),
            "beat": beats[i] if i < len(beats) else None,
        })
        at += delivered
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

    # STATIC LoRAs. The template carries ONE loader because an API export omits
    # bypassed nodes, so a request asking for more than one is asking for
    # wiring the template does not have -- say so rather than silently dropping
    # the extras.
    loras = models.get("loras") or []
    slots = roles_of(graph).get("h3.lora.static") or []
    if len(loras) > len(slots):
        raise TakeError(f"{len(loras)} static LoRA(s) requested but the template "
                        f"wires {len(slots)}. Schedule the rest through "
                        f"`lora_schedule`, which has no limit.")
    for nid, spec in zip(slots, loras):
        ins = graph[nid].setdefault("inputs", {})
        ins["lora_name"] = spec["name"] if isinstance(spec, dict) else spec
        if isinstance(spec, dict) and "strength" in spec:
            ins["strength_model"] = float(spec["strength"])

    refs = req.get("references") or []
    slots = sorted(r for r in roles_of(graph) if r.startswith("h3.ref."))
    if len(refs) > len(slots):
        raise TakeError(f"{len(refs)} reference(s) requested but the template "
                        f"wires {len(slots)}. A fourth reference has also been "
                        f"measured to break subject motion.")
    for role, name in zip(slots, refs):
        _set(graph, role, "image", name)

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
    for field in ("head", "beats", "tail", "subject_def_1", "retention_1",
                  "soundscape", "music", "task_type"):
        if field in req:
            _set(graph, "h3.links", field, req[field])

    for field, key in (("chunk_frames", "chunk_frames"),
                       ("chunk_count", "chunk_count")):
        if key in req:
            _set(graph, "h3.plan", field, int(req[key]))
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
    for i, row in enumerate(cast or []):
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
