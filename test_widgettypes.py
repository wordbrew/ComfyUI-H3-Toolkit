"""The widget-type check in normalize_workflows.py.

WHAT IT IS FOR
  On 2026-09-01 a script rewrote H3ScenePrompt's optional tail as ten empty
  strings, putting "" into three INT `pictures_N` widgets. Nothing catches that
  on load — the graph opens fine and fails later at prompt validation with "An
  input value has the wrong type", naming widget labels inherited from a
  different graph. Twenty minutes to work out that the labels were lying.

WHY BY TYPE AND NOT BY POSITION
  A positional check of widgets_values against saved files was tried on
  2026-08-29 and produced 29 findings, none real: the array carries no names
  and whether a converted widget keeps its slot varies between ComfyUI
  versions. The multiset of TYPES survives all of that — a node declaring seven
  strings and three ints cannot be holding ten strings however the slots line
  up — which is why this check is worth having where the order check was not.

    python3 test_widgettypes.py
"""
import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).parent.resolve()
_spec = importlib.util.spec_from_file_location(
    "h3norm", _root / "normalize_workflows.py")
_nw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nw)

widget_types, accepts, problems = (_nw.widget_types, _nw.accepts,
                                   _nw.widget_problems)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


# A stand-in for H3ScenePrompt's shape: strings, ints, and link-only inputs
# mixed together, because the link-only ones must not consume widget slots.
SPEC = {
    "required": {
        "task_type": (["video editing", "t2v"],),
        "seconds": ("INT", {"default": 5}),
        "style": ("STRING", {"multiline": True}),
        "shots": ("STRING", {"multiline": True}),
    },
    "optional": {
        "model": ("MODEL",),
        "subject_def_1": ("STRING", {}),
        "pictures_1": ("INT", {"default": 0}),
        "retention_1": ("STRING", {}),
        "scale": ("FLOAT", {"default": 1.0}),
        "enabled": ("BOOLEAN", {"default": True}),
    },
}


def node(values):
    return {"type": "T", "id": 1, "widgets_values": values}


# --- what counts as a widget ------------------------------------------------ #
print("link-only inputs are not widgets and never appear in widgets_values")
names = [n for n, _ in widget_types(SPEC)]
ok("MODEL is excluded", "model" not in names)
check("everything else, in declaration order", names,
      ["task_type", "seconds", "style", "shots", "subject_def_1",
       "pictures_1", "retention_1", "scale", "enabled"])
check("a choice list reads as a string slot",
      dict(widget_types(SPEC))["task_type"], "COMBO")

# --- the type rules --------------------------------------------------------- #
print("bool is ruled out before int, or every True looks like a legal INT")
ok("True is not an INT", not accepts("INT", True))
ok("True is a BOOLEAN", accepts("BOOLEAN", True))
ok("1 is not a BOOLEAN", not accepts("BOOLEAN", 1))
ok("an int satisfies FLOAT", accepts("FLOAT", 1))
ok("a float does not satisfy INT", not accepts("INT", 1.5))
ok("COMBO takes a string", accepts("COMBO", "video editing"))
ok("STRING does not take an int", not accepts("STRING", 3))

# --- the real case ---------------------------------------------------------- #
print("the bug this exists for: a string written into an INT widget")
good = ["video editing", 5, "sty", "shots", "", 0, "", 1.0, True]
surplus, unfilled = problems(node(good), SPEC)
check("a correct array is silent", surplus, [])

broken = ["video editing", 5, "sty", "shots", "", "", "", 1.0, True]
surplus, unfilled = problems(node(broken), SPEC)
check("one surplus string", len(surplus), 1)
check("and it names the INT slot, not where the leftover landed",
      [n for n, _ in unfilled], ["pictures_1"])

print("the whole-tail wipe that actually happened")
wiped = ["video editing", 5, "sty", "shots"] + [""] * 5
surplus, unfilled = problems(node(wiped), SPEC)
ok("surplus strings are reported", len(surplus) >= 1)
ok("and the INT slot is named", "pictures_1" in [n for n, _ in unfilled])

# --- what must NOT fire ----------------------------------------------------- #
print("no false alarms — this is the check the positional one failed")
ok("a short array is fine, widgets can be left unset",
   problems(node(["video editing", 5]), SPEC)[0] == [])
ok("a dict widgets_values (VHS nodes) is skipped",
   problems({"type": "T", "id": 1, "widgets_values": {"video": "a.mp4"}},
            SPEC)[0] == [])
ok("no widgets_values at all is fine",
   problems({"type": "T", "id": 1}, SPEC)[0] == [])
ok("None is skipped rather than counted as a bad value",
   problems(node(["video editing", 5, None, "shots", "", 0, "", 1.0, True]),
            SPEC)[0] == [])
ok("a node with no widgets at all is skipped",
   problems(node(["x"]), {"required": {"model": ("MODEL",)}})[0] == [])

# ORDER ambiguity is exactly what sank the positional check. A converted widget
# can leave its value out, shifting everything after it — the types still fit,
# so this must stay quiet.
# THE SEED TRAP, which the checker hit on its own first outing. The frontend
# inserts a control_after_generate widget after a seed and it is not in
# INPUT_TYPES, so the saved array is one longer than the declaration. Reporting
# that would mean a false alarm on every node carrying a seed.
print("a seed's control_after_generate widget is expected, not surplus")
SEEDED = {"required": {
    "seed": ("INT", {"default": 0}),
    "text": ("STRING", {}),
}}
names = [n for n, _ in widget_types(SEEDED)]
check("a synthetic control slot follows the seed", names,
      ["seed", "seed_control", "text"])
ok("an array WITH the control value is quiet",
   problems(node([2691, "fixed", "hello"]), SEEDED)[0] == [])
ok("and one without it is still quiet",
   problems(node([2691, "hello"]), SEEDED)[0] == [])
EXPLICIT = {"required": {
    "noise": ("INT", {"control_after_generate": True}),
    "text": ("STRING", {}),
}}
ok("core's explicit control_after_generate is honoured too",
   problems(node([7, "randomize", "hi"]), EXPLICIT)[0] == [])
ok("an INT not named seed gets no free slot",
   problems(node([7, "randomize", "hi"]),
            {"required": {"count": ("INT", {}), "text": ("STRING", {})}})[0] != [])

print("a shifted array whose types still fit stays quiet")
shifted = ["video editing", 5, "sty", "shots", "", 0, "", 1.0]
ok("one widget converted to an input", problems(node(shifted), SPEC)[0] == [])

print()
if fails:
    print(f"{len(fails)} failure(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("widget types: all checks pass")
