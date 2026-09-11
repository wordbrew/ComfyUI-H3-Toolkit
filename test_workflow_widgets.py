"""Every saved workflow's widget values line up with the node's real widgets.

THE BUG THIS CATCHES
  H3 09's `H3ScenePrompt` carried eight values for a node with seventeen widgets,
  and they were offset: `soundscape` held 640 and `non_diegetic_music` held 1120
  -- the canvas width and height, from a node that is not this one. So every
  render made from that workflow put "640" into the prompt as its soundtrack.

  Nothing errored. ComfyUI maps `widgets_values` to widgets by POSITION and
  coerces whatever it finds, so a short or shifted list is silently wrong rather
  than loud. The workflows in this pack are generated, and a generator writing
  the list by hand has no feedback at all -- which is how this survived.

  The companion trap is written up in `widget_slots`: a seed's
  `control_after_generate` takes the slot AFTER it, not one at the end.

  `test_slot_contract.py` answers a different question: does a node's declared
  order still match what saved graphs expect. This one answers whether the saved
  graphs were ever right in the first place.

WHAT COUNTS AS A WIDGET
  Every entry in INPUT_TYPES that is not a link-only socket. A widget CONVERTED
  to an input still occupies its slot in `widgets_values`, which is the detail
  that makes hand-authoring these lists error-prone.

    python3 test_workflow_widgets.py
"""
import glob
import importlib.util
import json
import pathlib
import sys
import types

_root = pathlib.Path(__file__).parent.resolve()
_pkg = types.ModuleType("h3wpk")
_pkg.__path__ = [str(_root)]
sys.modules["h3wpk"] = _pkg

# Socket-only types never take a widget slot.
SOCKET_TYPES = {"MODEL", "CLIP", "VAE", "IMAGE", "MASK", "LATENT", "AUDIO",
                "CONDITIONING", "NOISE", "GUIDER", "SAMPLER", "SIGMAS",
                "H3_CHUNK_PLAN", "H3_CROP_DATA", "VHS_BatchManager",
                "VHS_VIDEOINFO", "VHS_FILENAMES", "MODEL_PATCH", "VIDEO"}
# Nodes whose widget list genuinely varies per instance (autogrow templates).
SKIP = {"H3Script"}

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def load(name):
    spec = importlib.util.spec_from_file_location(f"h3wpk.{name}", _root / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"h3wpk.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# every module that registers nodes, loaded as a package so relative imports work
CLASSES = {}
for path in sorted(_root.glob("*.py")):
    if path.name.startswith("test_") or path.name == "__init__.py":
        continue
    try:
        mod = load(path.stem)
    except Exception:
        continue                       # needs ComfyUI; nothing to check here
    CLASSES.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {})

print(f"{len(CLASSES)} node class(es) importable without ComfyUI")


def widget_slots(cls):
    """[(name, type)] in the order `widgets_values` is read, or None."""
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return None
    out = []
    for group in ("required", "optional"):
        for name, decl in (spec.get(group) or {}).items():
            t = decl[0] if isinstance(decl, (list, tuple)) else decl
            cfg = decl[1] if isinstance(decl, (list, tuple)) and len(decl) > 1 else {}
            # `forceInput` makes an otherwise-widget entry a SOCKET, so it takes
            # no slot in widgets_values. Missing that reported H3ChunkLora's
            # `chunk_index` as holding the wrong value in a workflow that was
            # perfectly fine.
            if isinstance(cfg, dict) and cfg.get("forceInput"):
                continue
            if isinstance(t, list):
                out.append((name, "COMBO"))
            elif t in SOCKET_TYPES:
                continue
            else:
                out.append((name, t))
            # THE COMPANION SITS IN THE MIDDLE, NOT AT THE END. ComfyUI gives a
            # seed widget a `control_after_generate` of its own, and it takes the
            # NEXT slot -- so every value after the seed is one further along
            # than the declaration order suggests. Reading it as a trailing extra
            # made eleven correct workflows look offset, which is the mistake
            # this whole file exists to avoid making.
            if name in ("seed", "noise_seed"):
                out.append((name + "_control", "COMBO"))
    return out


def fits(value, kind):
    """Could this value plausibly be the saved state of that widget?"""
    if value is None:
        return True
    if kind == "COMBO":
        return isinstance(value, (str, int, float, bool))
    if kind == "BOOLEAN":
        return isinstance(value, bool)
    if kind in ("INT", "FLOAT"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "STRING":
        # THE TELL. A number sitting on a STRING widget is a value that slid in
        # from somewhere else -- H3 09 had the canvas width on `soundscape`.
        return isinstance(value, str)
    return True


files = sorted(glob.glob(str(_root / "workflows" / "*.json"))
               + glob.glob(str(_root / "workflows" / "legacy" / "*.json")))
print(f"checking {len(files)} workflow(s)\n")

seen = 0
for path in files:
    name = pathlib.Path(path).name
    try:
        graph = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        fails.append(f"{name}: not valid JSON ({exc})")
        continue
    for node in graph.get("nodes", []):
        cls = CLASSES.get(node.get("type"))
        if cls is None or node["type"] in SKIP:
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue                   # VHS-style dicts key by name, not position
        slots = widget_slots(cls)
        if slots is None:
            continue
        seen += 1

        # A SHORT LIST IS NOT A FAULT. Appending a widget is the supported way to
        # extend a node, and every graph saved before that append legitimately
        # carries one fewer value and takes the default. What is never right is a
        # value of the wrong SHAPE for the widget it lands on, because positional
        # mapping means everything after it is on the wrong widget too.
        if len(values) > len(slots) + 1:
            fails.append(f"{name}: {node['type']} (id {node['id']}) has "
                         f"{len(values)} values for {len(slots)} widgets")
            print(f"  FAIL {name}: {node['type']} id {node['id']} — too many values")
            continue
        for pos, value in enumerate(values[:len(slots)]):
            w_name, kind = slots[pos]
            if not fits(value, kind):
                fails.append(
                    f"{name}: {node['type']} (id {node['id']}) slot {pos} "
                    f"`{w_name}` is a {kind} but holds {value!r} — the values "
                    f"are offset, so this and every one after it is on the "
                    f"wrong widget")
                print(f"  FAIL {name}: {node['type']} id {node['id']} — "
                      f"`{w_name}` ({kind}) holds {value!r}")
                break

print(f"\n{seen} node instance(s) checked")
print(f"{len(fails)} failure(s)" if fails
      else "workflow widgets: every saved value lands on the widget it was written for")
sys.exit(1 if fails else 0)
