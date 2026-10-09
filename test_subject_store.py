#!/usr/bin/env python3
"""The subject store as a CONTRACT: anything may produce a subject.

A subject is a directory — card.json plus images/NNN.ext — and that is the
whole interface between whatever made it and everything that uses it. krea2, a
ComfyUI render, or a person with a folder of photos are interchangeable
producers, and keeping that true is what makes the dual path (Comfy OR a
standalone engine) possible rather than a rewrite.

Two things are tested here because both are load-bearing and neither is
obvious:

  ADDING is not SAVING. save_subject writes a whole subject; add_anchors
  extends one. A persistent companion needs the second — three more looks next
  week, keeping what is there.

  NUMBERING CONTINUES FROM THE HIGHEST FILE, never from the count. A subject
  whose 002 was deleted must not get a second 002, because an older take's
  <Picture 2> would then resolve to a different face.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")
import audit_nodes as A                                        # noqa: E402

A.load_pack()
ch = sys.modules["_h3audit.character"]

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")

fails = []


def ok(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


def check(label, got, want):
    ok(f"{label}: {got!r}", got == want)


tmp = tempfile.mkdtemp()
ch.characters_dir = lambda create=False: tmp


def subject(name, nums):
    d = os.path.join(tmp, name, "images")
    os.makedirs(d, exist_ok=True)
    for n in nums:
        with open(os.path.join(d, f"{n}.png"), "wb") as f:
            f.write(PNG)
    with open(os.path.join(tmp, name, "card.json"), "w") as f:
        json.dump({"name": name, "anchors": len(nums)}, f)


def blank(n="new.png"):
    p = os.path.join(tmp, n)
    with open(p, "wb") as f:
        f.write(PNG)
    return p


print("adding keeps what is there")
subject("lily", ["001", "002"])
card, added = ch.add_anchors("lily", [blank()])
check("the new one follows the last", [os.path.basename(a) for a in added],
      ["003.png"])
check("and the count is the files on disk", card["anchors"], 3)
ok("the originals survive",
   os.path.isfile(os.path.join(tmp, "lily", "images", "001.png")))

print("\nA GAP IS NOT REFILLED, because an older take cites these numbers")
subject("jen", ["001", "003"])
_, added = ch.add_anchors("jen", [blank("n2.png")])
check("004, not 002", [os.path.basename(a) for a in added], ["004.png"])

print("\nprovenance is recorded, and accumulates")
card, _ = ch.add_anchors("lily", [blank("n3.png")],
                         source={"generator": "krea2", "seed": 42})
check("the generator", card["source"]["generator"], "krea2")
check("one run so far", len(card["source"]["runs"]), 1)
card, _ = ch.add_anchors("lily", [blank("n4.png")],
                         source={"generator": "comfy", "seed": 43})
check("a second run is kept too", len(card["source"]["runs"]), 2)
check("and the latest generator wins the top-level field",
      card["source"]["generator"], "comfy")
ok("while the first run is still readable",
   card["source"]["runs"][0]["seed"] == 42)

print("\nand it refuses what it cannot do")
try:
    ch.add_anchors("nobody", [blank("n5.png")])
    ok("adding to a subject that does not exist raises", False)
except ValueError as e:
    ok(f"adding to a subject that does not exist raises — {e}", True)
_, added = ch.add_anchors("lily", ["/nope/missing.png"])
check("a missing file is skipped, not invented", added, [])

print("\na fresh card carries source too, so a GENERATED subject says so")
c = ch.card_of("x", source={"generator": "krea2", "companion_id": "abc"})
check("source survives card_of", c["source"]["companion_id"], "abc")
c2 = ch.card_of("y")
check("and defaults to empty rather than missing", c2["source"], {})

shutil.rmtree(tmp)
print()
if fails:
    print(f"FAIL — {len(fails)} check(s)")
    for f in fails[:8]:
        print("  " + f)
    sys.exit(1)
print("subject store: adding extends, numbering holds, provenance accumulates")
