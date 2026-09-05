"""H3 Semantic Bridge — the span decision, which is the whole point of the node.

The published adapter blends EVERY conditioning token. In ref2va the sequence is
`<Picture n>: [vision] ... <Audio n>: <prompt>`, and those reference tokens are a
distribution the adapter never saw — its author reports degraded lip-sync when
reference audio is included. So we blend the PROMPT SUFFIX only.

`span()` is torch-free on purpose: it is the decision that has to be right, and
no torch is installed in WSL, so anything tensor-shaped could not be tested here
at all. The four lines of blending arithmetic that follow it are UNVERIFIED
outside ComfyUI.

    python3 test_semantic.py
"""
import importlib.util
import pathlib
import sys
import types

# semantic.py imports torch at module level for the node; the span logic does
# not use it. Stub it so this file can exercise the decision on its own.
for name in ("torch", "torch.nn"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].Module = type("Module", (), {"__init__": lambda s: None})
sys.modules["torch.nn"].Linear = lambda *a, **k: None
sys.modules["torch.nn"].SiLU = lambda *a, **k: None

_spec = importlib.util.spec_from_file_location(
    "h3sem", pathlib.Path(__file__).parent / "semantic.py")
sem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sem)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


print("the prompt is a SUFFIX — everything before it is reference material")
lo, note = sem.span(12, 40)
check("28 reference tokens are left alone", lo, 28)
check("and the note says which", "last 12 of 40" in note, True)
check("it names what is protected", "28 reference token(s) untouched" in note,
      True)

print("one reference token is still worth protecting")
check("span 39 of 40", sem.span(39, 40)[0], 1)

print("degenerate spans return None so the caller falls back and REPORTS it")
for n, why in ((40, "prompt fills the sequence"), (41, "longer than the run"),
               (0, "tokenizer reported nothing"), (-3, "negative")):
    lo, note = sem.span(n, 40)
    check(f"{why} -> no mask", lo, None)
    check(f"{why} -> says why", "not a usable suffix" in note, True)

print("the percentage is of the whole sequence, not of the prompt")
check("10 of 40 reads 25%", "(25%)" in sem.span(10, 40)[1], True)

print("no off-by-one at the boundary")
check("span n of n+1 keeps exactly one reference token", sem.span(39, 40)[0], 1)
check("span 1 of 40 masks only the last token", sem.span(1, 40)[0], 39)

print()
print("NOT COVERED HERE, and it needs a real render: the blend itself, the")
print("adapter load, and whether the token count the CLIP reports lines up")
print("with the conditioning's own length once vision blocks have expanded.")
print()
print(f"{len(fails)} failure(s)" if fails else "semantic bridge: all checks pass")
sys.exit(1 if fails else 0)
