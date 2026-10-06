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
raises("a fourth reference", lambda: ta.build(
    dict(REQ, references=["a.png", "b.png", "c.png", "d.png"])), "reference")
raises("two static loras", lambda: ta.build(
    dict(REQ, models=dict(REQ["models"], loras=[{"name": "1"}, {"name": "2"}]))),
    "static LoRA")
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

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("take api: all checks pass")
