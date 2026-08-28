"""H3SeamCheck's join arithmetic — the part that decides where you look."""
import sys, types, importlib.util, pathlib

# torch is not installed in WSL, and `import torch.nn.functional as Fn` needs a
# real package to walk, so build one out of empty modules.
torch = types.ModuleType("torch"); torch.__path__ = []
nn = types.ModuleType("torch.nn"); nn.__path__ = []
fn = types.ModuleType("torch.nn.functional")
torch.nn = nn; nn.functional = fn
for name, mod in (("torch", torch), ("torch.nn", nn),
                  ("torch.nn.functional", fn)):
    sys.modules.setdefault(name, mod)
root = pathlib.Path("/home/cgree/projects/ComfyUI-H3-Toolkit")
spec = importlib.util.spec_from_file_location("h3p", root / "__init__.py",
                                              submodule_search_locations=[str(root)])
m = importlib.util.module_from_spec(spec); sys.modules["h3p"] = m
try:
    spec.loader.exec_module(m)
except Exception as e:
    print("(pack needs comfy for some nodes:", type(e).__name__, ")")
lf = importlib.import_module("h3p.longform")
cp = importlib.import_module("h3p.chunkplan")
J = lf.H3SeamCheck._joins
fails = []
def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print("  FAIL", label, got, want)
    else:
        print("  ok  ", label)

# V2V, 400 frames, chunk 90, context 22 -> the joins are where each chunk's kept
# frames begin in the JOINED clip
ch, _ = cp.plan(400, chunk_frames=90, context=22)
joins = [j for j, _ in J({"chunks": ch}, 400)]
kept = [c["end"] - c["keep_from"] for c in ch]
check("one join per seam", len(joins), len(ch) - 1)
check("joins are the cumulative kept lengths", joins,
      [sum(kept[:i + 1]) for i in range(len(ch) - 1)])
check("and with exact coverage they equal keep_from", joins,
      [c["keep_from"] for c in ch[1:]])
check("each join carries the chunk that STARTS there",
      [c["pin"] for _, c in J({"chunks": ch}, 400)], [22] * (len(ch) - 1))

# no context: joins still land where the chunks butt together
ch2, _ = cp.plan(400, chunk_frames=90, context=0)
check("no-context joins", [j for j, _ in J({"chunks": ch2}, 400)],
      [c["keep_from"] for c in ch2[1:]])

# a clip shorter than the plan says (someone trimmed it) drops joins past the end
check("joins past the end are dropped", [j for j, _ in J({"chunks": ch}, 100)], [90])
check("no plan means no joins", J(None, 400), [])
check("a single chunk has no seam", J({"chunks": ch[:1]}, 90), [])

print()
print("FAIL" if fails else "seam check: all join arithmetic passes")
sys.exit(1 if fails else 0)
