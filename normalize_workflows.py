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


def fix(path, classes, write):
    d = json.load(open(path, encoding="utf-8"))
    notes, changed = [], False
    for n in d["nodes"]:
        cls = classes.get(n["type"])
        if cls is None:
            continue
        try:
            t = cls.INPUT_TYPES()
        except Exception:
            continue
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
    return notes


def main(argv):
    write = "--write" in argv
    classes = pack_classes()
    if not classes:
        print("could not load the pack's node classes")
        return 1
    total = 0
    for path in sorted(glob.glob("workflows/*.json")):
        notes = fix(path, classes, write)
        if notes:
            total += len(notes)
            print(("fixed " if write else "MISMATCH ") + path.split("/")[-1])
            print("\n".join(notes))
    if not total:
        print("every saved input order matches its node's declaration")
    elif not write:
        print(f"\n{total} node(s) would lose links on reload. "
              f"Re-run with --write to fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
