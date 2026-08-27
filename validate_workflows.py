#!/usr/bin/env python3
"""Check the example workflows before they reach ComfyUI.

Every workflow bug this pack has shipped was structural and silent: ComfyUI drops
a bad connection on load rather than complaining, so the graph opens looking fine
and fails at queue time, or worse, runs with a node quietly unwired. The three we
actually shipped were slot indices left at 0, one link id feeding two consumers,
and links left behind after a node's outputs were renumbered.

Run it after editing any workflow JSON by hand or by script:

    python3 validate_workflows.py workflows/*.json
"""

import glob
import json
import sys
from collections import Counter


def check(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    nodes = {n["id"]: n for n in d.get("nodes", [])}
    links = d.get("links", [])
    errs = []

    ids = Counter(l[0] for l in links)
    for lid, c in ids.items():
        if c > 1:
            errs.append(f"link {lid} defined {c} times")

    # a link id may terminate at exactly one input; sharing one is the bug that
    # looks like it works until the second consumer silently loses its wire
    dests = Counter((l[3], l[4]) for l in links)
    for (nid, slot), c in dests.items():
        if c > 1:
            errs.append(f"node {nid} input slot {slot} is fed by {c} links")

    for lid, src, sslot, dst, dslot, ty in links:
        s, t = nodes.get(src), nodes.get(dst)
        if s is None:
            errs.append(f"link {lid}: source node {src} does not exist")
            continue
        if t is None:
            errs.append(f"link {lid}: dest node {dst} does not exist")
            continue
        souts = s.get("outputs") or []
        tins = t.get("inputs") or []
        if sslot >= len(souts):
            errs.append(f"link {lid}: {s['type']}#{src} has no output slot {sslot}")
        elif lid not in (souts[sslot].get("links") or []):
            errs.append(f"link {lid}: not listed on {s['type']}#{src} "
                        f"output '{souts[sslot].get('name')}'")
        if dslot >= len(tins):
            errs.append(f"link {lid}: {t['type']}#{dst} has no input slot {dslot}")
        else:
            got = tins[dslot].get("link")
            if got != lid:
                errs.append(f"link {lid}: {t['type']}#{dst} input "
                            f"'{tins[dslot].get('name')}' points at {got}")
            it = tins[dslot].get("type")
            # an input type may be a UNION — "IMAGE,MASK", "FLOAT,INT,BOOLEAN" —
            # so membership, not equality. Comparing these with != flagged five
            # perfectly good links in a real workflow.
            if it and ty and "*" not in (it, ty):
                accepted = {p.strip() for p in str(it).split(",")}
                offered = {p.strip() for p in str(ty).split(",")}
                if not (accepted & offered):
                    errs.append(f"link {lid}: {ty} into a {it} input "
                                f"({t['type']}#{dst}.{tins[dslot].get('name')})")

    declared = {l[0] for l in links}
    for n in d.get("nodes", []):
        for o in (n.get("outputs") or []):
            for lid in (o.get("links") or []):
                if lid not in declared:
                    errs.append(f"{n['type']}#{n['id']} output '{o.get('name')}' "
                                f"lists link {lid}, which is not defined")
        for inp in (n.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None and lid not in declared:
                errs.append(f"{n['type']}#{n['id']} input '{inp.get('name')}' "
                            f"points at link {lid}, which is not defined")

    if d.get("last_link_id") is not None and declared:
        if max(declared) > d["last_link_id"]:
            errs.append(f"last_link_id is {d['last_link_id']} but link "
                        f"{max(declared)} exists")
    return errs


def main(paths):
    bad = 0
    # expand patterns ourselves so `python3 validate_workflows.py` checks every
    # workflow whether or not the shell got to the argument first
    expanded = []
    for p in paths:
        expanded.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not expanded:
        print("no workflows matched")
        return 1
    for p in expanded:
        errs = check(p)
        name = p.rsplit("/", 1)[-1]
        if errs:
            bad += 1
            print(f"FAIL {name}")
            for e in errs:
                print(f"      {e}")
        else:
            print(f"ok   {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["workflows/*.json"]))
