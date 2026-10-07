"""Exercise H3MultiRefModLoader against the real refmods on disk.

Run from the ComfyUI root with its own interpreter, so storage and the vendor
bridge resolve exactly as they do in a render.
"""
import sys, types

sys.path.insert(0, ".")
sys.path.insert(0, "custom_nodes")

import importlib
pkg = importlib.import_module("ComfyUI-H3RefModLoader".replace("-", "_")) \
    if False else None  # the folder name is not an identifier; load by path instead

import importlib.util, pathlib
ROOT = pathlib.Path("custom_nodes/ComfyUI-H3RefModLoader").resolve()

# Load the folder as a package named h3rml so relative imports work.
spec = importlib.util.spec_from_file_location(
    "h3rml", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
mod = importlib.util.module_from_spec(spec)
sys.modules["h3rml"] = mod
spec.loader.exec_module(mod)

Multi = mod.NODE_CLASS_MAPPINGS["H3MultiRefModLoader"]
Simple = mod.NODE_CLASS_MAPPINGS["H3SimpleRefModLoader"]
print("registered:", sorted(mod.NODE_CLASS_MAPPINGS))

it = Multi.INPUT_TYPES()
opt = it["optional"]
slots = [k for k in opt if k.startswith("refmod_")]
print(f"slots: {len(slots)}  ({', '.join(sorted(slots))})")
names = [n for n in opt["refmod_1"][0] if n != "(none)"]
print(f"refmods on disk: {len(names)}")
for n in names:
    print("   ", n)

# --- a stand-in MODEL with the wrapper API the node checks for -------------- #
class FakeModel:
    def __init__(self):
        self.model_options = {}
        self._wrappers = {}
    def clone(self):
        c = FakeModel()
        c.model_options = dict(self.model_options)
        c._wrappers = {k: list(v) for k, v in self._wrappers.items()}
        return c
    def get_wrappers(self, kind, key):
        return self._wrappers.get((kind, key), [])
    def add_wrapper_with_key(self, kind, key, fn):
        self._wrappers.setdefault((kind, key), []).append(fn)
    def remove_wrappers_with_key(self, kind, key):
        self._wrappers.pop((kind, key), None)

node = Multi()
fails = []

def check(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
    if not cond:
        fails.append(label)

print("\n--- nothing selected passes through")
m, info = node.apply(FakeModel())
check("model returned, no wrapper installed", not m._wrappers)
check("info explains the passthrough", "no RefMods" in info)

print("\n--- one mod")
audio = next((n for n in names if "audio" in n.lower()), None)
visual = next((n for n in names if "audio" not in n.lower()), None)
print(f"  using audio={audio!r} visual={visual!r}")
m1, info1 = node.apply(FakeModel(), refmod_1=audio, strength_1=1.0)
check("wrapper installed", bool(m1._wrappers))
check("fingerprint set", "h3sr_refmods_v1" in m1.model_options)
print("  info:\n   " + info1.replace("\n", "\n   "))

print("\n--- TWO mods in one node (the point of this node)")
m2, info2 = node.apply(FakeModel(), refmod_1=visual, strength_1=1.0,
                       refmod_2=audio, strength_2=1.0)
check("wrapper installed", bool(m2._wrappers))
check("info reports 2 RefMods", info2.startswith("2 RefMod"))
check("both slots listed", "slot 1" in info2 and "slot 2" in info2)
check("fingerprint differs from the single-mod stack",
      m2.model_options["h3sr_refmods_v1"] != m1.model_options["h3sr_refmods_v1"])
print("  info:\n   " + info2.replace("\n", "\n   "))

print("\n--- the blocks actually reach the conditioning")
class FakeGuider:
    def __init__(self):
        self.conds = {"positive": [{"minimax_refs": [{"kind": "image", "pre": "existing"}]}]}
class FakeExecutor:
    def __init__(self, g): self.class_obj = g
    def __call__(self, *a, **k): return self.class_obj.conds
g = FakeGuider()
wrapper = m2._wrappers[list(m2._wrappers)[0]][0]
seen = wrapper(FakeExecutor(g))
refs = seen["positive"][0]["minimax_refs"]
check(f"{len(refs)} refs on the conditioning (1 pre-existing + 2 ours)", len(refs) == 3)
check("the pre-existing ref2va reference SURVIVES", refs[0].get("pre") == "existing")
check("ours are marked refmod", all(r.get("refmod") for r in refs[1:]))
kinds = [r["kind"] for r in refs[1:]]
check(f"kinds in slot order: {kinds}", len(kinds) == 2)
check("conds restored after the call", g.conds["positive"][0]["minimax_refs"][0].get("pre") == "existing"
      and len(g.conds["positive"][0]["minimax_refs"]) == 1)

print("\n--- order is coordinate order, so it changes identity")
m3, _ = node.apply(FakeModel(), refmod_1=audio, strength_1=1.0,
                   refmod_2=visual, strength_2=1.0)
check("swapped slots fingerprint differently",
      m3.model_options["h3sr_refmods_v1"] != m2.model_options["h3sr_refmods_v1"])

print("\n--- strength 0 skips a slot, not the whole stack")
m4, info4 = node.apply(FakeModel(), refmod_1=visual, strength_1=1.0,
                       refmod_2=audio, strength_2=0.0)
check("only 1 RefMod applied", info4.startswith("1 RefMod"))

print("\n--- bad input is refused")
for label, kw in (("strength above 1", {"refmod_1": audio, "strength_1": 1.5}),
                  ("strength NaN", {"refmod_1": audio, "strength_1": float("nan")}),
                  ("boolean strength", {"refmod_1": audio, "strength_1": True})):
    try:
        node.apply(FakeModel(), **kw)
        check(label + " rejected", False)
    except ValueError:
        check(label + " rejected", True)

print("\n--- a second loader on the same line is refused, not silently dropped")
try:
    node.apply(m2, refmod_1=audio, strength_1=1.0)
    check("second loader refused", False)
except ValueError as e:
    check("second loader refused", "already has a RefMod bridge" in str(e))

print("\n--- the Simple loader still works unchanged")
s, = Simple().apply(FakeModel(), audio, 1.0, True)
check("Simple loader attaches", bool(s._wrappers))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
