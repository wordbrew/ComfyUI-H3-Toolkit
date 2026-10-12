#!/usr/bin/env python3
"""A fully-loaded composed graph, validated the way ComfyUI validates it.

CJ's Comfy run proved the NODES and the MODEL. What it cannot prove is that
takeapi builds the same graph — the loaders, the depth chain, the gates, the
audio slots and the numbering are minted here, not drawn by hand.

So this composes one request with every feature on at once and checks what
core checks at submit: that every class exists, every link points at a real
node and a real output slot, every declared input is satisfied, and no
placeholder survives. That is the check that was missing when H3ChunkRef
shipped asking for `H3CHUNKPLAN` — the graph composed fine and only failed at
submit, after a restart, with a message about two strings differing by two
underscores.
"""
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import audit_nodes as A                                        # noqa: E402

classes, failed = A.load_pack()
assert not failed, failed

sys.modules.setdefault("torch", types.ModuleType("torch"))
pkg = types.ModuleType("h3t"); pkg.__path__ = [str(ROOT)]
sys.modules["h3t"] = pkg


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

#: Core nodes this pack's templates mint or reference. Their schemas are not
#: importable here, so they are checked for EXISTENCE in the graph's wiring
#: rather than against their declarations — core does the rest at submit.
CORE = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
    "LoadImage", "LoadAudio", "LoadVideo", "GetVideoComponents",
    "ImageFromBatch", "CreateVideo", "SaveVideo", "VAEDecode",
    "VAEDecodeAudio", "SamplerCustomAdvanced", "KSamplerSelect",
    "BasicGuider", "RandomNoise", "ModelSamplingMiniMaxH3",
    "MiniMaxH3ReferenceToVideo", "ResolutionSelector",
    "DepthAnything_V2", "DownloadAndLoadDepthAnythingV2Model",
    "Model Attention Backend", "ModelAttentionBackend",
    # the sigma schedule: both verified present in this ComfyUI --
    # ManualSigmas in comfy_extras/nodes_custom_sampler.py,
    # MiniMaxH3SigmaShift in comfy_extras/nodes_minimax_h3.py
    "ManualSigmas", "MiniMaxH3SigmaShift",
}

#: Output counts for the core nodes this pack MINTS, so a slot index into one
#: is range-checked too. Without these the check silently skipped every core
#: node — a wrong slot on a LoadAudio composed fine and only failed at submit.
#: Each was read out of this ComfyUI's source, not assumed:
#:   LoadAudio           comfy_extras/nodes_audio.py   outputs=[IO.Audio.Output()]
#:   LoadImage           nodes.py                      RETURN_TYPES = ("IMAGE","MASK")
#:   LoadVideo           comfy_extras/nodes_video.py   [Video]
#:   GetVideoComponents  comfy_extras/nodes_video.py   [Image, Audio, Float, Combo, Combo]
#:   ImageFromBatch      comfy_extras/nodes_images.py  outputs=[IO.Image.Output()]
#:   DepthAnything_V2    custom_nodes/.../nodes.py     RETURN_TYPES = ("IMAGE",)
CORE_OUTPUTS = {
    "LoadAudio": 1, "LoadImage": 2, "LoadVideo": 1,
    "GetVideoComponents": 5, "ImageFromBatch": 1,
    "DepthAnything_V2": 1, "DownloadAndLoadDepthAnythingV2Model": 1,
}

REQ = {
    "models": {"unet": "H3/model.safetensors", "clip": "enc.safetensors",
               "vae": "H3/video_vae.safetensors",
               "audio_vae": "H3/audio_vae.safetensors",
               "loras": [{"name": "h3/a.safetensors", "strength": 0.8},
                         {"name": "h3/b.safetensors", "strength": 0.4}]},
    "cast": [{"key": "lily", "describe": "a woman with red hair",
              "pictures": 2},
             {"key": "room", "setting": "a living room"},
             {"key": "lamp", "subkind": "thing", "describe": "a brass lamp"}],
    "references": ["face_a.png", "face_b.png"],
    "audio_references": ["h3_subjects/lily/voice.wav"],
    "chunk_references": [
        {"source": "plate.png", "kind": "image", "role": "composition",
         "chunks": "1-2"},
        {"source": "open.png", "kind": "image", "role": "frame 0"},
        {"source": "move.mp4", "kind": "video",
         "role": "composition and motion", "spans": "48 96 the push",
         "with_audio": True, "audio_marker": "reference"},
        {"source": "room.wav", "kind": "audio", "marker": "weak_reference"},
    ],
    "head": "@lily in @room", "beats": "b1\nb2\nb3", "tail": "T",
    "chunk_count": 3, "context": 39,
    "scenes": [{"frames": 192},
               {"frames": 141, "continuity": "carry"},
               {"frames": 192, "continuity": "cut"}],
    "lora_schedule": "2 | h3/x.safetensors | 1.0",
    "seed": 4242, "filename_prefix": "H3/validate",
    "megapixels": 0.5, "aspect_ratio": "2:3 (Portrait Photo)",
}

fails = []


def ok(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


print("composing one request with every feature on at once")
graph = ta.build(REQ)
print(f"  {len(graph)} nodes")

print("\nevery class is one that exists")
unknown = sorted({n["class_type"] for n in graph.values()
                  if n["class_type"] not in classes
                  and n["class_type"] not in CORE})
ok(f"no unknown class{' — ' + ', '.join(unknown) if unknown else ''}",
   not unknown)

print("\nevery link points at a real node and a real output slot")
bad_links = []
for nid, node in graph.items():
    for field, v in (node.get("inputs") or {}).items():
        if not (isinstance(v, list) and len(v) == 2
                and isinstance(v[0], str)):
            continue
        tgt, slot = v
        if tgt not in graph:
            bad_links.append(f"{node['class_type']}.{field} -> missing {tgt}")
            continue
        src = graph[tgt]["class_type"]
        cls = classes.get(src)
        n_out = (len(getattr(cls, "RETURN_TYPES", ()) or ()) if cls is not None
                 else CORE_OUTPUTS.get(src, 0))
        if True:
            if n_out and not (0 <= slot < n_out):
                bad_links.append(
                    f"{node['class_type']}.{field} -> {src}[{slot}] "
                    f"but {src} has {n_out} output(s)")
for b in bad_links:
    print("       " + b)
ok(f"{len(bad_links)} broken link(s)", not bad_links)

print("\nevery REQUIRED input of our own nodes is satisfied")
missing = []
for nid, node in graph.items():
    cls = classes.get(node["class_type"])
    if cls is None or not hasattr(cls, "INPUT_TYPES"):
        continue
    have = set((node.get("inputs") or {}).keys())
    for field, decl in (cls.INPUT_TYPES().get("required") or {}).items():
        if field in have:
            continue
        # an autogrow/templated input is satisfied by its prefixed members
        if any(k.startswith(field + ".") for k in have):
            continue
        missing.append(f"{node['class_type']}.{field}")
for m in missing:
    print("       " + m)
ok(f"{len(missing)} unsatisfied required input(s)", not missing)

print("\nand the request actually reached the graph")
ok("no placeholder survives", ta.placeholders_in(graph) == [])
_cond = ta._one(graph, "h3.conditioning")
_ins = graph[_cond]["inputs"]
ok("pictures wired", any(k.startswith("ref_images.") for k in _ins))
ok("a clip wired", any(k.startswith("ref_videos.") for k in _ins))
ok("its soundtrack wired",
   any(k.startswith("ref_video_audios.") for k in _ins))
ok("two standalone sounds wired",
   len([k for k in _ins if k.startswith("ref_audios.")]) == 2)
ok("both static LoRAs chained",
   len([1 for n in graph.values()
        if n["class_type"] == "LoraLoaderModelOnly"]) == 2)
ok("one shared depth model",
   len([1 for n in graph.values()
        if n["class_type"] == "DownloadAndLoadDepthAnythingV2Model"]) == 1)
_links = ta._one(graph, "h3.links")
ok("@handles resolved to numbers",
   "<Subject 1>" in str(graph[_links]["inputs"]["head"])
   and "@" not in str(graph[_links]["inputs"]["head"]))
ok("retention_1 is wired to a gate",
   isinstance(graph[_links]["inputs"].get("retention_1"), list))

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("composed graph: structurally what ComfyUI expects")
