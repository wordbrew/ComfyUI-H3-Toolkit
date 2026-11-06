#!/usr/bin/env python3
"""The number in a citation must be the number core will present.

CJ: "im prompting for stuff and finding that frame 0 isn't always keeping as
frame 0 when i am using reference material... it'll like start as a reference
image itself, then morph over."

It was our bug. The slot an image was wired into and the number its clause
cited came from two different counters, so with three cast anchors present a
`frame 0` reference sitting in `ref_image_4` was cited as `<Picture 1>` -- the
first cast anchor. The prompt said "cast anchor A is the first frame of the
video; the opening frame matches it exactly", and the model obliged: it opened
on a reference photo and then travelled to the real shot.

CORE'S RULE, read from comfy_extras/nodes_minimax_h3.py, is three loops that
each skip what is not there:

    for img in (ref_images or {}).values():
        if img is None: continue                       -> <Picture N>
    for video in (ref_videos or {}).values():
        if video is None: continue
        if soundtrack is not None: emit an audio item  -> <Audio j> BEFORE
        emit the video item                            -> <Video k>
    for audio in (ref_audios or {}).values():
        if audio is None: continue                     -> <Audio j>

So a citation is the position among the references ACTUALLY SENT, a dark gate
renumbers everything after it, and a clip's soundtrack is labelled immediately
before its own clip. None of that is knowable when the graph is built.

This test reimplements that rule as `present()` -- deliberately from core's
source, not from our builder -- composes real graphs, executes the gate chain
the way ComfyUI would, and checks the clauses name what `present()` says they
should. A test written off our own numbering would have passed throughout.
"""
import importlib.util
import pathlib
import re
import shutil
import sys
import types

for _pc in pathlib.Path(__file__).resolve().parent.rglob("__pycache__"):
    shutil.rmtree(_pc, ignore_errors=True)

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import audit_nodes as A                                        # noqa: E402

_classes, _failed = A.load_pack()
assert not _failed, _failed
sys.modules.setdefault("torch", types.ModuleType("torch"))
_pkg = types.ModuleType("h3t"); _pkg.__path__ = [str(ROOT)]
sys.modules["h3t"] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(f"h3t.{name}",
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"h3t.{name}"] = m
    spec.loader.exec_module(m)
    return m


for _m in ("timing", "chunkplan", "prompt_lint", "character", "h3script",
           "chunklora", "chunkref"):
    _load(_m)
ta = _load("takeapi")
cr = sys.modules["h3t.chunkref"]

fails = []


def ck(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(label)


MODELS = {"unet": "u", "clip": "c", "vae": "v", "audio_vae": "av"}


def compose(references=(), chunk_references=(), voices=(), count=2):
    req = {"chunk_frames": 192, "chunk_count": count, "context": 39,
           "beats": "\n".join(f"beat {i + 1}" for i in range(count)),
           "prompt": "p", "models": dict(MODELS),
           "references": list(references),
           "chunk_references": [dict(r) for r in chunk_references]}
    if voices:
        req["audio_references"] = list(voices)
    return ta.build(req)


def cond_inputs(graph):
    nid = [k for k, v in graph.items()
           if (v.get("_meta") or {}).get("title") == "h3.conditioning"][0]
    return graph[nid]["inputs"]


def slots(ins, prefix):
    """The wirings of one reference family, in slot order."""
    keys = [k for k in ins if k.startswith(prefix)]
    return [ins[k] for k in sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1]))]


def present(graph, chunk_index, total):
    """CORE'S numbering, for the chunk being rendered. -> {node id: citation}

    Walks the three families exactly as nodes_minimax_h3.py does, deciding
    emptiness the way the graph will at runtime: a gate is dark when its
    scope excludes this chunk, and the reference sampler is empty unless this
    chunk is a reference_sample join (never, in these fixtures).
    """
    ins = cond_inputs(graph)
    titles = {}
    for role, nids in ta.roles_of(graph).items():
        for nid in nids:
            titles[nid] = role

    def dark(link):
        nid = link[0]
        role = titles.get(nid, "")
        if role == "h3.refsample":
            return True                      # None off a join
        if not role.startswith("h3.chunkref."):
            return False                     # a plain loader is always live
        gin = graph[nid]["inputs"]
        live, _ = cr.active_here(gin.get("chunks"), chunk_index, total)
        if gin.get("role") == "frame 0":
            plan = []                        # refusal needs the plan; see below
        return not live

    out, pics = {}, 0
    for link in slots(ins, "ref_images.ref_image_"):
        if dark(link):
            continue
        pics += 1
        out[link[0]] = f"<Picture {pics}>"
    vids, auds = 0, 0
    sound = {}
    for k, v in ins.items():
        if k.startswith("ref_video_audios.ref_video_audio_"):
            sound[int(k.rsplit("_", 1)[1])] = v
    for i, link in enumerate(slots(ins, "ref_videos.ref_video_")):
        if dark(link):
            continue
        if i in sound:
            auds += 1
            out[("sound", link[0])] = f"<Audio {auds}>"
        vids += 1
        out[link[0]] = f"<Video {vids}>"
    for link in slots(ins, "ref_audios.ref_audio_"):
        if dark(link):
            continue
        auds += 1
        out.setdefault(("voice", link[0]), f"<Audio {auds}>")
    return out


def run_chain(graph, chunk_index, total):
    """Execute the gate chain in order. -> {node id: its retention text}

    ComfyUI walks the chain because each gate's `tally_in` comes from the one
    before it; this does the same by hand so the test does not need ComfyUI.
    """
    gates = sorted((r for r in ta.roles_of(graph)
                    if r.startswith("h3.chunkref.")),
                   key=lambda r: int(r.rsplit(".", 1)[1]))
    node = cr.H3ChunkRef()
    tally, said = None, {}
    for role in gates:
        nid = ta.roles_of(graph)[role][0]
        gin = dict(graph[nid]["inputs"])
        plan = None
        if isinstance(gin.get("chunk_plan"), list):
            plan = {"chunks": PLAN_CHUNKS}
        kwargs = {
            "role": gin.get("role"), "chunks": gin.get("chunks"),
            "chunk_index": chunk_index, "image": "IMG",
            "ordinal": gin.get("ordinal", 1), "chunk_plan": plan,
            "label": gin.get("label", ""),
            "marker": gin.get("marker", cr.DEFAULT_MARKER),
            "kind": gin.get("kind", "image"),
            "how": gin.get("how", ""),
            "with_audio": gin.get("with_audio", False),
            "audio_marker": gin.get("audio_marker", "reference"),
            "cast_images": gin.get("cast_images", 0),
            "tally_in": tally,
        }
        res = node.go(**kwargs)["result"]
        said[nid] = res[2]
        tally = res[4]
    return said, tally


PLAN_CHUNKS = []


def cites(text):
    """Every <Tag N> a retention text names, in order."""
    return re.findall(r"<(?:Picture|Video|Audio) \d+>", text or "")


# ---------------------------------------------------------------- the bug
print("THE REPORTED BUG: a frame-0 image behind a cast")
g = compose(references=["cast/a.png", "cast/b.png", "cast/c.png"],
            chunk_references=[{"source": "open.png", "kind": "image",
                              "role": "frame 0", "chunks": "all",
                              "label": "the opening still"}])
ins = cond_inputs(g)
order = [(ta.roles_of(g), l) for l in slots(ins, "ref_images.ref_image_")]
titles = {}
for role, nids in ta.roles_of(g).items():
    for nid in nids:
        titles[nid] = role
ck("the cast holds the first slots, then the gate, then the sampler",
   [titles[l[0]] for l in slots(ins, "ref_images.ref_image_")],
   ["h3.ref.0", "h3.ref.1", "h3.ref.2", "h3.chunkref.0", "h3.refsample"])

want = present(g, 0, 2)
gate_id = ta.roles_of(g)["h3.chunkref.0"][0]
ck("core will present that image as the fourth picture", want[gate_id],
   "<Picture 4>")
said, _ = run_chain(g, 0, 2)
ck("AND THAT IS WHAT THE CLAUSE SAYS", cites(said[gate_id]), ["<Picture 4>"])

# ---------------------------------------------------------- no cast at all
print("\nand the off-by-one with no cast, which the sampler used to cause")
g = compose(chunk_references=[{"source": "open.png", "kind": "image",
                              "role": "frame 0", "chunks": "all"}])
gate_id = ta.roles_of(g)["h3.chunkref.0"][0]
want = present(g, 0, 2)
said, _ = run_chain(g, 0, 2)
ck("the sampler is last, so the image really is the first picture",
   want[gate_id], "<Picture 1>")
ck("and is cited as such", cites(said[gate_id]), ["<Picture 1>"])

# ------------------------------------------------- a dark gate renumbers
print("\na gate that is dark on this chunk renumbers the ones after it")
PLAN_CHUNKS = [{}, {}, {}]
g = compose(references=["cast/a.png"],
            chunk_references=[
                {"source": "one.png", "kind": "image", "role": "composition",
                 "chunks": "1"},
                {"source": "two.png", "kind": "image", "role": "reference",
                 "chunks": "all", "marker": "partially_preserved"},
            ], count=3)
a = ta.roles_of(g)["h3.chunkref.0"][0]
b = ta.roles_of(g)["h3.chunkref.1"][0]
for chunk, want_a, want_b in ((0, "<Picture 2>", "<Picture 3>"),
                              (1, None, "<Picture 2>")):
    expect = present(g, chunk, 3)
    said, _ = run_chain(g, chunk, 3)
    ck(f"chunk {chunk + 1}: core presents the gated one as {want_a}",
       expect.get(a), want_a)
    ck(f"chunk {chunk + 1}: and the clause agrees",
       cites(said[a]), [want_a] if want_a else [])
    ck(f"chunk {chunk + 1}: the one after it moves to {want_b}",
       expect.get(b), want_b)
    ck(f"chunk {chunk + 1}: and ITS clause agrees too",
       cites(said[b]), [want_b])

# --------------------------------------------------------- audio ordering
print("\na clip's soundtrack is labelled before its clip, and voices follow")
PLAN_CHUNKS = [{}, {}]
g = compose(references=["cast/a.png"],
            chunk_references=[
                {"source": "move.mp4", "kind": "video",
                 "role": "composition and motion", "chunks": "all",
                 "with_audio": True, "audio_marker": "reference"},
            ],
            voices=["lily.wav"])
vid = ta.roles_of(g)["h3.chunkref.0"][0]
expect = present(g, 0, 2)
ck("the clip is the first video", expect.get(vid), "<Video 1>")
ck("its soundtrack takes Audio 1", expect.get(("sound", vid)), "<Audio 1>")
voice_links = slots(cond_inputs(g), "ref_audios.ref_audio_")
ck("one voice is wired", len(voice_links), 1)
ck("AND THE VOICE IS AUDIO 2, not Audio 1",
   expect.get(("voice", voice_links[0][0])), "<Audio 2>")
said, _ = run_chain(g, 0, 2)
ck("the gate cites its clip and its sound, in core's order",
   cites(said[vid]), ["<Video 1>", "<Audio 1>", "<Video 1>"])

# --------------------------------- a sounded clip cannot be gated silently
print("\nand a clip that sends its sound is not allowed to be gated")
import json                                                    # noqa: E402
tmpl = json.loads(json.dumps(ta.load_template(ta.DEFAULT_TEMPLATE)))
_gate, lint, tracks = ta.set_chunk_references(tmpl, [
    {"source": "move.mp4", "kind": "video", "role": "composition and motion",
     "chunks": "2", "with_audio": True}])
ck("it is reported rather than silently renumbering every voice",
   any("spans the whole take" in m for m in lint), True)
gate_role = [r for r in ta.roles_of(tmpl) if r.startswith("h3.chunkref.")][0]
ck("and its scope is widened to the whole take",
   tmpl[ta.roles_of(tmpl)[gate_role][0]]["inputs"]["chunks"], "all")

# ---------------------------------------------- the free-text `how`
print("\nthe plain `reference` role says what you tell it to")
PLAN_CHUNKS = [{}, {}]
g = compose(chunk_references=[
    {"source": "body.png", "kind": "image", "role": "reference",
     "chunks": "all", "marker": "partially_preserved",
     "how": "take the body proportions and anatomy, not the face"}])
gid = ta.roles_of(g)["h3.chunkref.0"][0]
said, _ = run_chain(g, 0, 2)
ck("the marker leads", said[gid].split(" - ")[0].endswith("partially_preserved"),
   True)
ck("the sentence is the user's",
   "take the body proportions and anatomy, not the face." in said[gid], True)
ck("nothing about framing was added",
   "composition" not in said[gid] and "frame 0" not in said[gid], True)

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:10]:
        print("  " + f)
    sys.exit(1)
print("ref numbering: every citation names what core will present")
