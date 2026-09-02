#!/usr/bin/env python3
"""Put every saved node's input slots back in the order its node DECLARES.

WHY THIS EXISTS
  A link stores a slot INDEX, and on load ComfyUI reconciles the saved input
  array against the node's current INPUT_TYPES. If the two disagree the links
  do not error -- they are silently dropped, and the node comes up with the
  wires missing. That happened to H3 Chunk Close's `audio` on 2026-08-28: it
  had been appended to the END of the saved array, which is the right rule for
  OUTPUTS and WIDGETS and the wrong one for inputs.

  Outputs and widgets_values are positional and a new one must be appended.
  Inputs are reconciled by name against the declaration, so the saved order has
  to MATCH the declaration. Two rules, and they are not the same rule.

    python3 normalize_workflows.py            # report only
    python3 normalize_workflows.py --write    # fix in place

WIDGET order is NOT checked here. widgets_values carries no names, and whether a
widget converted to an input keeps its slot varies between ComfyUI versions, so
a positional check against saved files produces false alarms rather than
findings -- tried on 2026-08-29, 29 "problems" of which none were real.
test_slot_contract.py pins the declared order on the NODE side instead, which is
where the mistake actually gets made.

WIDGET TYPES ARE CHECKED, and that is a different question with a different
answer. On 2026-09-01 a script rewrote H3ScenePrompt's optional tail as ten
empty strings, putting "" into the three INT `pictures_N` widgets. Nothing
load-checks that: it queued, and failed at prompt validation with "An input
value has the wrong type" against widget labels left over from a different
graph, which explained nothing.

The check is by TYPE MULTISET, never by position. A node declaring seven string
widgets and three ints cannot be holding ten strings however the slots line up,
so the finding survives every ordering ambiguity that sank the positional check.
Values are matched greedily to declared widget types -- same index first, so a
well-ordered graph matches trivially -- and only values fitting NO remaining
slot are reported. `--write` does not repair these: the correct value is a
judgement call, so it reports and leaves them.
"""

import glob
import importlib
import importlib.util
import json
import pathlib
import sys
import types


def _stub_torch():
    """The pack imports torch at module level; the slot schema does not need it."""
    torch = types.ModuleType("torch"); torch.__path__ = []
    nn = types.ModuleType("torch.nn"); nn.__path__ = []
    fn = types.ModuleType("torch.nn.functional")
    torch.nn = nn; nn.functional = fn
    for name, mod in (("torch", torch), ("torch.nn", nn),
                      ("torch.nn.functional", fn)):
        sys.modules.setdefault(name, mod)


def pack_classes():
    _stub_torch()
    root = pathlib.Path(__file__).parent.resolve()
    spec = importlib.util.spec_from_file_location(
        "h3pack_slots", root / "__init__.py",
        submodule_search_locations=[str(root)])
    pk = importlib.util.module_from_spec(spec)
    sys.modules["h3pack_slots"] = pk
    try:
        spec.loader.exec_module(pk)
    except Exception:
        pass                      # some nodes need ComfyUI; the rest still load
    out = {}
    for name in ("audio", "budget", "character", "chunkrun", "crop", "longform",
                 "mask", "prompt_lint", "prompt_links", "prompt_rewriter",
                 "prompt_scene", "video", "windowing"):
        try:
            mod = importlib.import_module("h3pack_slots." + name)
        except Exception:
            continue
        out.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {})
    return out


def widget_types(spec):
    """Declared widget slots as (name, kind), skipping link-only inputs.

    A widget is an input whose type is primitive or a COMBO (a list of
    choices). MODEL, IMAGE, LATENT and friends are links and never appear in
    widgets_values.
    """
    out = []
    for name, decl in list(spec.get("required", {}).items()) + \
            list(spec.get("optional", {}).items()):
        t = decl[0] if isinstance(decl, (tuple, list)) and decl else decl
        if isinstance(t, (list, tuple)):
            out.append((name, "COMBO"))            # a choice list -> a string
        elif t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            out.append((name, t))
    return out


def accepts(kind, value):
    """Would ComfyUI take `value` in a widget of this kind?"""
    # bool is a subclass of int, so it has to be ruled out first or every
    # True/False looks like a legal INT and the check misses real swaps
    if isinstance(value, bool):
        return kind == "BOOLEAN"
    if kind == "BOOLEAN":
        return False
    if kind == "INT":
        return isinstance(value, int)
    if kind == "FLOAT":
        return isinstance(value, (int, float))     # 1 is a fine float
    return isinstance(value, str)                  # STRING, COMBO


def widget_problems(node, spec):
    """(surplus values, unfilled slots) when the types cannot all be placed.

    Greedy, and by TYPE not position: try the same index first so an ordinary
    graph matches one-to-one, then any unused slot. What survives is a value the
    node could not accept under ANY arrangement of its widgets.

    Reports the UNFILLED SLOTS rather than where the leftovers landed. Greedy
    matching consumes the compatible slots first, so the surplus always washes
    up at the end of the array and naming those indices points at innocent
    widgets -- which is precisely the unhelpful thing ComfyUI's own validation
    error does, and the reason this check exists.
    """
    saved = node.get("widgets_values")
    if not isinstance(saved, list):
        return [], []             # some nodes store a dict; not our business
    slots = widget_types(spec)
    if not slots:
        return [], []
    used, surplus = set(), []
    for idx, value in enumerate(saved):
        if value is None:
            continue
        if idx < len(slots) and idx not in used and accepts(slots[idx][1], value):
            used.add(idx)
            continue
        hit = next((k for k, (_, kind) in enumerate(slots)
                    if k not in used and accepts(kind, value)), None)
        if hit is None:
            surplus.append(value)
        else:
            used.add(hit)
    if not surplus:
        return [], []
    unfilled = [(name, kind) for k, (name, kind) in enumerate(slots)
                if k not in used]
    return surplus, unfilled


def fix(path, classes, write):
    d = json.load(open(path, encoding="utf-8"))
    notes, types_bad, changed = [], [], False
    for n in d["nodes"]:
        cls = classes.get(n["type"])
        if cls is None:
            continue
        try:
            t = cls.INPUT_TYPES()
        except Exception:
            continue
        surplus, unfilled = widget_problems(n, t)
        if surplus:
            kinds = sorted({type(v).__name__ for v in surplus})
            slots = ", ".join(f"{nm} ({k})" for nm, k in unfilled) or "none"
            types_bad.append(
                f"    {n['type']}#{n['id']}: {len(surplus)} value(s) of type "
                f"{'/'.join(kinds)} fit no widget; unfilled slots: {slots}")
        decl = list(t.get("required", {})) + list(t.get("optional", {}))
        saved = n.get("inputs") or []
        names = [i["name"] for i in saved]
        known = [x for x in names if x in decl]
        want = [x for x in decl if x in names]
        if known == want:
            continue
        notes.append(f"    {n['type']}#{n['id']}: {known} -> {want}")
        changed = True
        if not write:
            continue
        # keep anything the pack does not declare (a converted widget from an
        # older build) where it is, relative to the end
        by_name = {i["name"]: i for i in saved}
        rest = [i for i in saved if i["name"] not in decl]
        n["inputs"] = [by_name[x] for x in want] + rest
        pos = {i["name"]: k for k, i in enumerate(n["inputs"])}
        for i in n["inputs"]:
            if i.get("link") is None:
                continue
            for l in d["links"]:
                if l[0] == i["link"] and l[3] == n["id"]:
                    l[4] = pos[i["name"]]
    if changed and write:
        json.dump(d, open(path, "w", encoding="utf-8"), indent=2)
    return notes, types_bad


def main(argv):
    write = "--write" in argv
    classes = pack_classes()
    if not classes:
        print("could not load the pack's node classes")
        return 1
    total, bad_types = 0, 0
    for path in sorted(glob.glob("workflows/*.json")):
        notes, types_bad = fix(path, classes, write)
        if notes:
            total += len(notes)
            print(("fixed " if write else "MISMATCH ") + path.split("/")[-1])
            print("\n".join(notes))
        if types_bad:
            bad_types += len(types_bad)
            print("WRONG WIDGET TYPE " + path.split("/")[-1])
            print("\n".join(types_bad))
    if not total:
        print("every saved input order matches its node's declaration")
    elif not write:
        print(f"\n{total} node(s) would lose links on reload. "
              f"Re-run with --write to fix.")
    if bad_types:
        # not repairable here: the right value is a judgement call, and
        # guessing one is how a graph renders something nobody asked for
        print(f"\n{bad_types} node(s) hold a value their widgets cannot accept. "
              f"These fail at PROMPT VALIDATION, not on load, and the error "
              f"names stale labels — fix them by hand.")
        return 2
    if not total:
        print("every widget value fits a declared slot")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
