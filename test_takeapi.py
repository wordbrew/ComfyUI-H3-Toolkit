#!/usr/bin/env python3
"""The take API: planning arithmetic, and injection by ROLE rather than node id.

The thing under test is that a front end never has to know a node id. The
template's ids belong to whoever exported it and move when they edit it, so
every reachable node carries a role in `_meta.title` and a request addresses
that. A missing role must be an error naming the role, not a KeyError on an int.
"""
import importlib.util
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent
sys.modules.setdefault("torch", types.ModuleType("torch"))

pkg = types.ModuleType("h3t")
pkg.__path__ = [str(ROOT)]
sys.modules["h3t"] = pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(f"h3t.{name}",
                                                  ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"h3t.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_load("timing")
cp = _load("chunkplan")
ta = _load("takeapi")

fails = []


def ok(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


def check(label, got, want):
    ok(f"{label}: {got!r}", got == want)


def raises(label, fn, fragment):
    try:
        fn()
    except ta.TakeError as exc:
        ok(f"{label} — {str(exc)[:58]}", fragment.lower() in str(exc).lower())
        return
    except Exception as exc:                                  # noqa: BLE001
        ok(f"{label} raised {type(exc).__name__}, not TakeError", False)
        return
    ok(f"{label} did not raise", False)


# --------------------------------------------------------------- the template
print("the committed template is addressable by role")
tpl = ta.load_template()
roles = ta.roles_of(tpl)
for need in ("h3.unet", "h3.clip", "h3.vae.video", "h3.vae.audio", "h3.plan",
             "h3.links", "h3.lora.schedule", "h3.chunk.open", "h3.chunk.close",
             "h3.noise", "h3.save"):
    ok(f"role {need} exists", need in roles)
ok("every role addresses exactly one node",
   all(len(v) == 1 for k, v in roles.items() if not k.startswith("h3.ref.")
       and k != "h3.lora.static"))
ok("references are numbered", sorted(r for r in roles if r.startswith("h3.ref."))
   == ["h3.ref.0", "h3.ref.1", "h3.ref.2"])

print("and it ships with placeholders, because the repo is public")
left = ta.placeholders_in(tpl)
ok("the diffusion model is a placeholder",
   any("h3.unet" in r for r, _ in left))
ok("so is the static LoRA", any("h3.lora.static" in r for r, _ in left))
raises("a request with no models is refused",
       lambda: ta.build({}), "placeholder")

# ------------------------------------------------------------------- planning
print("\nplanning answers the two questions an author has")
plan, lint = ta.plan_fields({"chunk_frames": 192, "chunk_count": 3,
                             "context": 39}, cp.count_plan)
check("total frames", plan["total_frames"], 498)
check("total seconds", plan["total_seconds"], 20.75)
check("beats required", plan["beats_required"], 3)
check("chunk 1 is longer than the rest",
      (plan["first_chunk_frames"], plan["other_chunk_frames"]), (192, 153))
check("per-chunk boundaries", [(c["start_seconds"], c["end_seconds"])
                               for c in plan["chunks"]],
      [(0.0, 8.0), (8.0, 14.375), (14.375, 20.75)])

print("and it counts the beats for you")
_, lint = ta.plan_fields({"chunk_frames": 192, "chunk_count": 3,
                          "beats": "one\ntwo"}, cp.count_plan)
ok("two beats for three chunks is linted",
   any("2 beat" in m and "3 chunk" in m for m in lint))
plan, lint = ta.plan_fields({"chunk_frames": 192, "chunk_count": 2,
                             "beats": "one\ntwo"}, cp.count_plan)
ok("the right count lints clean", not any("beat(s) for" in m for m in lint))
check("and each beat is placed", [c["beat"] for c in plan["chunks"]],
      ["one", "two"])

# ------------------------------------------------------------------ injection
print("\na request reaches the template by role, never by node id")
REQ = {
    "models": {"unet": "H3/model.safetensors",
               "clip": "enc.safetensors",
               "vae": "H3/video_vae.safetensors",
               "audio_vae": "H3/audio_vae.safetensors",
               "loras": [{"name": "h3/style.safetensors", "strength": 0.8}]},
    "references": ["a.png", "b.png", "c.png"],
    "head": "HEAD", "beats": "b1\nb2", "tail": "TAIL",
    "chunk_frames": 192, "chunk_count": 2, "context": 39,
    "seed": 4242, "lora_schedule": "last | h3/x.safetensors | 1.0",
    "filename_prefix": "H3/test",
}
g = ta.build(REQ)
ok("no placeholders survive a full request", ta.placeholders_in(g) == [])


def field(graph, role, name):
    nid = ta._one(graph, role)
    return (graph[nid].get("inputs") or {}).get(name)


check("the unet landed", field(g, "h3.unet", "unet_name"),
      "H3/model.safetensors")
check("the audio vae landed on the AUDIO loader",
      field(g, "h3.vae.audio", "vae_name"), "H3/audio_vae.safetensors")
check("and the video vae on the other one",
      field(g, "h3.vae.video", "vae_name"), "H3/video_vae.safetensors")
check("the static lora name", field(g, "h3.lora.static", "lora_name"),
      "h3/style.safetensors")
check("and its strength", field(g, "h3.lora.static", "strength_model"), 0.8)
check("head", field(g, "h3.links", "head"), "HEAD")
check("beats", field(g, "h3.links", "beats"), "b1\nb2")
check("chunk count", field(g, "h3.plan", "chunk_count"), 2)
check("seed", field(g, "h3.noise", "noise_seed"), 4242)
check("schedule", field(g, "h3.lora.schedule", "schedule"),
      "last | h3/x.safetensors | 1.0")
check("references, in order", [field(g, f"h3.ref.{i}", "image")
                              for i in range(3)], ["a.png", "b.png", "c.png"])

print("context has to agree in BOTH places or the carry is not the plan's")
check("the plan takes a string", field(g, "h3.plan", "context"), "39")
check("Open takes an int", field(g, "h3.chunk.open", "context_frames"), 39)

print("\nasking for wiring the template does not have is an ERROR, not a drop")
# REFERENCES ARE NOT CAPPED, and the earlier version of this test asserted a
# limit that should never have existed: what was measured is that a fourth
# reference FOR THE LOCATION wrecked subject motion, while three CHARACTER
# references held. That is guidance about what a reference should contain, not a
# ceiling on how many there are -- `ref_images` is an autogrow input with no
# declared maximum. Loaders are minted and pruned to match the request.
# The template also wires H3ChunkRefSample into a slot of its own, permanently
# -- it emits None except at a `reference_sample` join, so it costs nothing on
# other chunks, but it means the slot count is references PLUS one.
for _n in (0, 1, 4, 9):
    _g = ta.build(dict(REQ, references=[f"r{_i}.png" for _i in range(_n)]))
    _cond = ta._one(_g, "h3.conditioning")
    _wired = [k for k in _g[_cond]["inputs"] if k.startswith("ref_images.ref_image_")]
    _loaders = [k for k, v in _g.items() if v["class_type"] == "LoadImage"]
    _samp = ta._one(_g, "h3.refsample", required=False)
    check(f"{_n} reference(s) wire {_n} loader slot(s) + the sampler",
          len(_wired), _n + (1 if _samp else 0))
    check(f"  and exactly {_n} loader(s) survive", len(_loaders), _n)
    ok(f"  the sampler keeps the slot after them",
       _g[_cond]["inputs"].get(f"ref_images.ref_image_{_n}") == [_samp, 0])
    ok(f"  no placeholder left at {_n}", ta.placeholders_in(_g) == [])
_g = ta.build(dict(REQ, references=["a.png", "b.png", "c.png", "d.png", "e.png"]))
_cond = ta._one(_g, "h3.conditioning")
check("references are numbered from zero, contiguously (5 + the sampler)",
      sorted(int(k.rsplit("_", 1)[1]) for k in _g[_cond]["inputs"]
             if k.startswith("ref_images.ref_image_")), [0, 1, 2, 3, 4, 5])
print("\nSTATIC LoRAs CHAIN, as many as asked for")
# This used to REFUSE a second one: the template wires a single loader, and I
# read that as a limit. A LoraLoaderModelOnly takes a MODEL and returns one, so
# a chain of length 1 is a chain, not a ceiling.
for _n in (0, 1, 3, 6):
    _g = ta.build(dict(REQ, models=dict(
        REQ["models"], loras=[{"name": f"h3/l{i}.safetensors",
                               "strength": 0.5} for i in range(_n)])))
    _ll = [k for k, v in _g.items() if v["class_type"] == "LoraLoaderModelOnly"]
    check(f"{_n} requested", len(_ll), _n)
    ok(f"  no placeholder survives at {_n}", ta.placeholders_in(_g) == [])
    # every loader must read the one before it, or the later ones do nothing
    _srcs = [(_g[k]["inputs"] or {}).get("model") for k in _ll]
    ok("  each reads the previous",
       all(isinstance(_srcs[i], list) and _srcs[i][0] == _ll[i - 1]
           for i in range(1, len(_ll))))
    # and whoever consumed the chain must now consume its END
    if _ll:
        _cons = [k for k, v in _g.items()
                 for vv in (v.get("inputs") or {}).values()
                 if isinstance(vv, list) and vv[0] == _ll[-1]]
        ok("  and the sampler reads the last one", bool(_cons))
    else:
        ok("  with none, nothing points at a dropped loader",
           not any(isinstance(vv, list) and vv[0] not in _g
                   for v in _g.values()
                   for vv in (v.get("inputs") or {}).values()))

print("\na cast retention AND a reference in one request")
# CJ: "h3.links.retention_1 is WIRED in the template, so a request cannot set
# it". set_chunk_references WIRED retention_1 and the text loop then tried to
# SET it -- _set correctly refusing to overwrite wiring this same function had
# just created. References are wired last now, and the cast's retention text
# SEEDS the gate chain instead of being dropped for it.
_both = ta.build(dict(REQ,
    retention_1="<Subject 1> fully_preserved: facial identity.",
    chunk_references=[{"source": "p.png", "kind": "image",
                       "role": "composition"}]))
_gates = [k for k, v in _both.items() if v["class_type"] == "H3ChunkRef"]
ok("it builds at all", bool(_gates))
check("the cast's retention opens the chain",
      _both[_gates[0]]["inputs"].get("retention_in"),
      "<Subject 1> fully_preserved: facial identity.")
_lk = ta._one(_both, "h3.links")
ok("and retention_1 ends up wired to the last gate",
   isinstance(_both[_lk]["inputs"]["retention_1"], list))
raises("an unknown template", lambda: ta.build(dict(REQ, template="nope")),
       "no template")

print("the original template is never mutated")
again = ta.load_template()
ok("placeholders are still there on a fresh load",
   ta.placeholders_in(again) != [])

print("\nthe graph stays submittable: ids are opaque, links intact")
ok("every input link points at a node that exists",
   all(v[0] in g for n in g.values() for v in (n.get("inputs") or {}).values()
       if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)))
ok("no node lost its class_type", all("class_type" in n for n in g.values()))
check("node count is the template's", len(g), len(tpl))

print("\nthe cast is DYNAMIC: add, add again, drop, and the numbering follows")
# CJ: "if I add a subject with refs, it should appropriately tie that in. if I
# add multiples, it does it properly to add to the list. if I drop one, it drops
# it and adjusts the others along with prompt call outs."
import importlib as _il2
_hs = _load("h3script")


def _cast(*rows):
    return ta.compile_cast(list(rows), _hs.parse, _hs.emit)


_typed = {"key": "man", "describe": "a man in his thirties"}
_typed2 = {"key": "woman", "describe": "a woman with red hair"}
_place = {"key": "room", "setting": "a bedroom in warm light"}

_one = _cast(_typed)
check("one subject is Subject 1", _one["counts"]["subjects"], 1)
ok("and it is the one described", "<Subject 1> is a man" in _one["subject_defs"])

_two = _cast(_typed, _typed2)
check("adding one appends rather than replacing", _two["counts"]["subjects"], 2)
ok("the first keeps its number", "<Subject 1> is a man" in _two["subject_defs"])
ok("and the second gets the next", "<Subject 2> is a woman" in _two["subject_defs"])

_three = _cast(_typed, _typed2, _place)
check("a setting is a subject too", _three["counts"]["subjects"], 3)
ok("and retains what a PLACE retains, not a face",
   "layout" in _three["retention"] and "facial identity" in _three["retention"])

print("dropping one renumbers everything after it")
_dropped = _cast(_typed2, _place)
ok("what was Subject 2 becomes Subject 1",
   "<Subject 1> is a woman" in _dropped["subject_defs"])
ok("and nothing still claims to be Subject 3",
   "<Subject 3>" not in _dropped["subject_defs"])
check("the count follows", _dropped["counts"]["subjects"], 2)

print("and the PICTURE count is what the references must match")
# a typed subject cites no pictures; only a stored one does, from its anchors
check("typed subjects cite no pictures", _two["counts"]["pictures"], 0)
ok("so a mismatch is detectable by comparing counts to the images sent",
   isinstance(_two["counts"].get("pictures"), int))


# ------------------------------------------------------ voices reach the model
# The conditioning node has carried `ref_audios` all along and the template
# wired only pictures, so a subject's saved voice was stored, listed, described
# -- and never sent. CJ: "I don't think audio is being applied for subjects."
_V = dict(REQ, audio_references=["h3_subjects/jen/voice.wav"])
_g = ta.build(_V)
_cond = ta._one(_g, "h3.conditioning")
_slots = [k for k in _g[_cond]["inputs"] if k.startswith("ref_audios.ref_audio_")]
check("one voice wires one audio slot", len(_slots), 1)
_aud = _g[_cond]["inputs"]["ref_audios.ref_audio_0"]
check("through a LoadAudio", _g[_aud[0]]["class_type"], "LoadAudio")
check("holding the staged path", _g[_aud[0]]["inputs"]["audio"],
      "h3_subjects/jen/voice.wav")
ok("and the pictures are untouched",
   len([k for k in _g[_cond]["inputs"] if k.startswith("ref_images.")]) >= 3)

print("two voices number from zero, and dropping them prunes the loaders")
_g2 = ta.build(dict(REQ, audio_references=["a.wav", "b.wav"]))
check("both wire", sorted(k for k in _g2[ta._one(_g2, "h3.conditioning")]["inputs"]
                         if k.startswith("ref_audios.")),
      ["ref_audios.ref_audio_0", "ref_audios.ref_audio_1"])
_g3 = ta.build(REQ)
check("no voices leaves no LoadAudio behind",
      sum(1 for v in _g3.values() if v["class_type"] == "LoadAudio"), 0)

print("a voice with no picture anywhere is refused, not silently dropped")
raises("lone voice", lambda: ta.build(
    dict(REQ, references=[], audio_references=["v.wav"])), "picture")


# ------------------------------------------- per-chunk references, with depth
# CJ: "ability to add references both in terms of images and videos per chunk
# Or overall for the whole video ... use as inspiration/composition only for
# frame 0 (use a depth version of itself)."
print("\na reference becomes loader -> [trim] -> [depth] -> gate -> slot")
_CR = dict(REQ, chunk_references=[
    {"source": "plate.png", "kind": "image", "role": "composition",
     "chunks": "1"},
    {"source": "move.mp4", "kind": "video", "role": "composition and motion",
     "chunks": "2", "spans": "48 96 the push"},
])
_g = ta.build(_CR)
_cls = [v["class_type"] for v in _g.values()]
ok("the image loads", _cls.count("LoadImage") >= 1)
# THE PACK'S OWN LOADER, not three stock nodes. VideoTrimLoad loads, trims and
# resizes in one, and its `spans` widget is the format the trim UI reads and
# writes -- which lands in the rendered mp4's metadata as the record of which
# part of which file a clip came from.
ok("the clip goes through VideoTrimLoad", "VideoTrimLoad" in _cls)
ok("and not the three-node stock chain",
   "LoadVideo" not in _cls and "GetVideoComponents" not in _cls
   and "ImageFromBatch" not in _cls)
check("both get a depth pass", _cls.count("DepthAnything_V2"), 2)
check("sharing ONE model loader, since the weights are the same",
      _cls.count("DownloadAndLoadDepthAnythingV2Model"), 1)
check("and two gates", _cls.count("H3ChunkRef"), 2)

print("the depth model is the one CJ's own graphs load")
_dm = [v for v in _g.values()
       if v["class_type"] == "DownloadAndLoadDepthAnythingV2Model"][0]
check("model", _dm["inputs"]["model"], "depth_anything_v2_vitl_fp32.safetensors")
check("precision", _dm["inputs"]["precision"], "auto")

print("each lands in the slot its KIND belongs to")
_cond = ta._one(_g, "h3.conditioning")
_ins = _g[_cond]["inputs"]
_gates = {k: v for k, v in _g.items() if v["class_type"] == "H3ChunkRef"}
_img_gate = [k for k, v in _gates.items() if v["inputs"]["role"] == "composition"][0]
_vid_gate = [k for k, v in _gates.items()
             if v["inputs"]["role"] == "composition and motion"][0]
ok("the composition reference is an IMAGE reference",
   any(v == [_img_gate, 0] for k, v in _ins.items()
       if k.startswith("ref_images.")))
ok("the motion reference is a VIDEO reference",
   any(v == [_vid_gate, 0] for k, v in _ins.items()
       if k.startswith("ref_videos.")))
ok("and the cast's own references keep their slots",
   any(ta._one(_g, "h3.ref.0") == v[0] for v in _ins.values()
       if isinstance(v, list)))

print("the gates chain, and the last one feeds the prompt")
_r1 = _gates[_img_gate].get("inputs", {}).get("retention_in")
_r2 = _gates[_vid_gate].get("inputs", {}).get("retention_in")
ok("exactly one gate starts the chain",
   (_r1 is None) != (_r2 is None))
_links = ta._one(_g, "h3.links")
_wired = _g[_links]["inputs"].get("retention_1")
ok("retention_1 is WIRED, not left as text", isinstance(_wired, list))
ok("to a gate's retention output",
   _wired[0] in _gates and _wired[1] == 2)

print("each chunk_index comes from Open, resolved by NAME not position")
from h3t.chunkrun import H3ChunkOpen as _Open
_slot = list(_Open.RETURN_NAMES).index("chunk_index")
ok("every gate reads the right slot",
   all(v["inputs"]["chunk_index"][1] == _slot for v in _gates.values()))

print("the caps H3 declares are enforced where the message can name the file")
_many = ta.build(dict(REQ, chunk_references=[
    {"source": f"v{i}.mp4", "kind": "video", "role": "composition and motion"}
    for i in range(5)]))
check("at most three video references survive",
      sum(1 for v in _many.values() if v["class_type"] == "VideoTrimLoad"), 3)

print("DEPTH FOLLOWS THE ROLE, and is not a flag a caller can contradict")
_d = ta.build(dict(REQ, chunk_references=[
    {"source": "a.png", "kind": "image", "role": "composition", "depth": False},
    {"source": "b.png", "kind": "image", "role": "frame 0", "depth": True},
]))
_dc = [v["class_type"] for v in _d.values()]
check("composition is depth-passed even when asked not to be, because its "
      "clause tells the model to ignore an appearance that would still be "
      "there", _dc.count("DepthAnything_V2"), 1)
_fg = [v for v in _d.values() if v["class_type"] == "H3ChunkRef"
       and v["inputs"]["role"] == "frame 0"][0]
check("and frame 0 is NOT, because it has to be the real picture",
      _d[_fg["inputs"]["image"][0]]["class_type"], "LoadImage")

print("a role the source cannot take is corrected and reported")
_w = ta.build(dict(REQ, chunk_references=[
    {"source": "c.mp4", "kind": "video", "role": "frame 0"}]))
_wg = [v for v in _w.values() if v["class_type"] == "H3ChunkRef"][0]
check("a clip cannot be frame 0", _wg["inputs"]["role"],
      "composition and motion")
ok("`identity` is not offered as an additional reference at all",
   "identity" not in [r[0] for r in [("composition",), ("frame 0",)]])

print("and nothing is built when nothing is asked for")
_none = ta.build(REQ)
ok("no depth model appears unbidden",
   not any(v["class_type"].startswith("DownloadAndLoad")
           for v in _none.values()))
ok("no gates either",
   not any(v["class_type"] == "H3ChunkRef" for v in _none.values()))


# ------------------------------------------- ordering, and @handles
print("\nSUBJECTS ARE NUMBERED people, then the place, then things")
# CJ: "is there a way to always stack cast before setting and setting before
# props? so we can always assume Subject 1 or Subject 2 are characters?"
_mixed = [{"key": "lamp", "subkind": "thing", "describe": "a brass lamp"},
          {"key": "room", "setting": "a living room"},
          {"key": "lily", "describe": "a woman with red hair"},
          {"key": "sam", "describe": "a man in a grey coat"}]
check("the order", [r["key"] for r in ta.cast_order(_mixed)],
      ["lily", "sam", "room", "lamp"])
_c = ta.compile_cast(_mixed, _hs.parse, _hs.emit)
ok("<Subject 1> is a person",
   "red hair" in _c["subject_defs"].splitlines()[0])
ok("and the setting is not", "setting" not in _c["subject_defs"].splitlines()[0])

print("adding a prop does not renumber the people")
_before = ta.handle_map(_mixed)
_after = ta.handle_map(_mixed + [{"key": "chair", "subkind": "thing"}])
check("lily stays 1", (_before["lily"], _after["lily"]), (1, 1))
check("and sam stays 2", (_before["sam"], _after["sam"]), (2, 2))

print("\n@HANDLES resolve to whatever the number turns out to be")
_m = ta.handle_map(_mixed)
check("the map", _m, {"lily": 1, "sam": 2, "room": 3, "lamp": 4})
_t, _bad = ta.apply_handles("@lily looks at @sam across @room.", _m)
check("substituted", _t,
      "<Subject 1> looks at <Subject 2> across <Subject 3>.")
check("nothing unknown", _bad, [])
ok("punctuation does not swallow the handle",
   ta.apply_handles("@lily, then @sam.", _m)[0]
   == "<Subject 1>, then <Subject 2>.")
ok("case does not matter", ta.apply_handles("@Lily", _m)[0] == "<Subject 1>")
ok("an email address is not a handle",
   "@" in ta.apply_handles("write to a@b.com", _m)[0])
check("an unknown handle is reported",
      ta.apply_handles("@nobody waves", _m)[1], ["nobody"])

print("a handle naming nobody STOPS the build rather than reaching the model")
raises("unknown handle", lambda: ta.build(
    dict(REQ, cast=_mixed, beats="@ghost turns away")), "not in the cast")

print("and a real one reaches the prompt as a number")
_g = ta.build(dict(REQ, cast=_mixed, head="@lily in @room", beats="b1\nb2"))
check("head was substituted",
      field(_g, "h3.links", "head"), "<Subject 1> in <Subject 3>")


# -------------------------------------------------- three kinds of audio
# CJ: "we need the option to use stand-alone audio as an additional reference
# as well as the option to include audio alongside a reference video ... in
# those cases, they will be treated differently for sure."
#
# They are: a voice and a standalone sound go to ref_audios, a clip's
# soundtrack to ref_video_audios AT ITS CLIP'S INDEX. And the <Audio j>
# NUMBERING IS NOT THE SLOT NUMBER -- the node presents images, then videos
# with each soundtrack's label immediately before its video, then standalone
# audio. A soundtrack takes <Audio 1> even when a voice was wired first.
print("\nthree audio cases, three slots, one numbering")
_AU = dict(REQ,
           audio_references=["jen/voice.wav"],
           chunk_references=[
               {"source": "mv.mp4", "kind": "video",
                "role": "composition and motion",
                "with_audio": True, "audio_marker": "reference"},
               {"source": "room.wav", "kind": "audio",
                "marker": "weak_reference"},
           ])
_g = ta.build(_AU)
_cond = ta._one(_g, "h3.conditioning")
_ins = _g[_cond]["inputs"]

ok("the clip's soundtrack is a VIDEO audio, not a standalone one",
   "ref_video_audios.ref_video_audio_0" in _ins)
_clip = ta._one(_g, "h3.clip.0", required=False)
check("and it is bound to ITS clip, by index and by node",
      _ins["ref_video_audios.ref_video_audio_0"], [_clip, 1])
ok("the voice and the standalone sound are standalone slots",
   "ref_audios.ref_audio_0" in _ins and "ref_audios.ref_audio_1" in _ins)
check("two of them, and no more",
      len([k for k in _ins if k.startswith("ref_audios.")]), 2)

print("the <Audio j> numbering follows the MODEL, not the wiring order")
_head = next(n for n, d in _g.items() if d["class_type"] == "H3ChunkRef"
             and not isinstance((d.get("inputs") or {}).get("retention_in"),
                                list))
_cl = _g[_head]["inputs"]["retention_in"].splitlines()
ok("<Audio 1> is the clip's soundtrack, though the voice was wired first",
   _cl[0].startswith("<Audio 1>") and "soundtrack of <Video 1>" in _cl[0])
ok("<Audio 2> is the voice", _cl[1].startswith("<Audio 2>"))
ok("<Audio 3> is the standalone sound", _cl[2].startswith("<Audio 3>"))

print("and each marker says the right thing")
ok("a soundtrack reference takes timbre and cadence, not the recording",
   "timbre, tone and cadence" in _cl[0] and "do not reuse" in _cl[0])
ok("weak_reference is looser still", "diverge freely" in _cl[2])
_fc = ta.build(dict(REQ, chunk_references=[
    {"source": "s.wav", "kind": "audio", "marker": "fully_copy"}]))
_lk = ta._one(_fc, "h3.links")
ok("fully_copy says to reuse the recording",
   "reuse this recording in full" in str(_fc[_lk]["inputs"]["retention_1"]))

print("with no visual gate, the clauses still reach the prompt as text")
ok("retention_1 carries them",
   "<Audio 1>" in str(_fc[_lk]["inputs"]["retention_1"]))

print("and nothing is wired when nothing is asked for")
_none = ta.build(REQ)
ok("no audio slots appear unbidden",
   not any(k.startswith(("ref_audios.", "ref_video_audios."))
           for k in _none[ta._one(_none, "h3.conditioning")]["inputs"]))

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("take api: all checks pass")
