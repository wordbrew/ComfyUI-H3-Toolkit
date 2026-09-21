"""The node contract — the shape saved workflows depend on.

WHAT THIS IS DEFENDING

  A saved workflow stores link endpoints by SLOT INDEX, not by name. Insert an
  output anywhere but the end and every graph built before it silently rewires
  to the wrong sockets -- it loads, it runs, and it is wrong.

  The two nodes here share `load_span` and must therefore share an output shape:
  a batch node whose sockets do not line up with the single node's is a node you
  cannot swap in, which is the only reason to have two.

  `OUTPUT_IS_LIST` gets its own check because its arity is separate from
  RETURN_TYPES and nothing validates them against each other. Too short and
  ComfyUI treats the missing outputs as scalars, which fails somewhere else
  entirely.

    python3 test_videotrim_nodes.py
"""
import importlib.util
import pathlib
import sys
import types

_root = pathlib.Path(__file__).parent.resolve()

# loader imports torch and ComfyUI at module level; neither is needed to read a
# class attribute, so stub what the import touches and nothing more
for _name in ("torch", "folder_paths", "comfy", "comfy.utils"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

_pkg = types.ModuleType("vt")
_pkg.__path__ = [str(_root)]
sys.modules["vt"] = _pkg
for _m in ("videoframes", "videotrim"):
    _spec = importlib.util.spec_from_file_location(f"vt.{_m}", _root / f"{_m}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"vt.{_m}"] = _mod
    _spec.loader.exec_module(_mod)

L = sys.modules["vt.videotrim"]

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


ONE, ALL = L.VideoTrimLoad, L.VideoTrimLoadAll

print("the outputs, in the order saved graphs store as slot indices")
check("names", ONE.RETURN_NAMES,
      ("images", "audio", "frame_count", "fps", "width", "height", "info"))
check("types", ONE.RETURN_TYPES,
      ("IMAGE", "AUDIO", "INT", "FLOAT", "INT", "INT", "STRING"))
check("one type per name", len(ONE.RETURN_TYPES), len(ONE.RETURN_NAMES))
# images and audio FIRST, because those are what a VHS graph was wired to and
# swapping this node in should be moving two wires, not seven
check("images then audio lead", ONE.RETURN_NAMES[:2], ("images", "audio"))
# deliberately NOT VHS_VIDEOINFO: that type belongs to VHS and emitting it
# would make this pack refuse to load without VHS installed
ok("no borrowed types", not any("VHS" in t for t in ONE.RETURN_TYPES))

print("the batch node is swappable, which is the only reason it exists")
check("same names", ALL.RETURN_NAMES, ONE.RETURN_NAMES)
check("same types", ALL.RETURN_TYPES, ONE.RETURN_TYPES)
# ARITY IS SEPARATE and nothing validates it: too short and ComfyUI reads the
# missing outputs as scalars, which fails somewhere else entirely
check("OUTPUT_IS_LIST covers every output",
      len(ALL.OUTPUT_IS_LIST), len(ALL.RETURN_TYPES))
ok("and every one of them is a list", all(ALL.OUTPUT_IS_LIST))
ok("the single node emits no lists", not hasattr(ONE, "OUTPUT_IS_LIST"))

print("inputs: the batch node drops span_index, because it emits them all")
one_in = ONE.INPUT_TYPES()
all_in = ALL.INPUT_TYPES()
ok("the single node picks one span", "span_index" in one_in["required"])
ok("the batch node does not", "span_index" not in all_in["required"])
check("everything else is shared",
      [k for k in one_in["required"] if k != "span_index"],
      list(all_in["required"]))
check("and the optionals are identical",
      list(one_in["optional"]), list(all_in["optional"]))

print("the widgets the timeline reads and writes are all present")
for name in ("file", "spans", "force_rate", "select_every_nth"):
    ok(f"{name} exists", name in one_in["required"])
# THE BUTTON THAT COULD NOT BE CLICKED. `video_upload` draws ComfyUI's own
# "choose file to upload" on the CANVAS at this widget's slot, and the timeline
# is a DOM element positioned over the node -- so the panel covers it and the
# button does nothing. Uploading lives in the panel, where the press lands on
# something that can receive it.
ok("no canvas upload button under the panel",
   "video_upload" not in (one_in["required"]["file"][1] or {}))
ok("nor on the batch node",
   "video_upload" not in (all_in["required"]["file"][1] or {}))
ok("but the tooltip says where uploading actually is",
   "Choose video" in one_in["required"]["file"][1]["tooltip"])

check("spans is multiline, because it holds one per line",
      one_in["required"]["spans"][1].get("multiline"), True)
# a fresh node must be usable before anything is dragged: 0 0 is "all of it"
check("the default span is the whole clip",
      one_in["required"]["spans"][1].get("default"), "0 0")
check("select_every_nth cannot be zero",
      one_in["required"]["select_every_nth"][1].get("min"), 1)
check("force_rate 0 is allowed, and means keep the source's own",
      one_in["required"]["force_rate"][1].get("min"), 0.0)

print("both nodes are registered, and named for what they do")
check("two nodes", sorted(L.NODE_CLASS_MAPPINGS), ["VideoTrimLoad",
                                                   "VideoTrimLoadAll"])
check("every class has a display name",
      sorted(L.NODE_DISPLAY_NAME_MAPPINGS), sorted(L.NODE_CLASS_MAPPINGS))
ok("the batch node says so in its name",
   "all spans" in L.NODE_DISPLAY_NAME_MAPPINGS["VideoTrimLoadAll"].lower())
# the cost of OUTPUT_IS_LIST is not obvious from the graph, so the description
# has to carry it
ok("and its description warns that the graph below runs N times",
   "once per span" in ALL.DESCRIPTION)

print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    sys.exit(1)
print("nodes: one output shape, two ways to spend it")
