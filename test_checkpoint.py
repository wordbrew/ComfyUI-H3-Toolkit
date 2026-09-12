"""The plan fingerprint — what decides whether a resume is allowed.

WHY REFUSE RATHER THAN WARN
  A checkpoint holds chunk N's sampler output so chunk N+1 can copy its tail. If
  the cuts move between saving and resuming, that tail belongs to a chunk that no
  longer starts where it did, and the seam it produces looks like the model's
  fault. That is the worst kind of bug to ship: a wrong result with no error.

  So the fingerprint covers exactly the fields that decide where a chunk starts
  and how much it carries, and nothing else. Re-wording the plan's info text must
  NOT invalidate a checkpoint, or the gate becomes too annoying to use and gets
  turned off.

    python3 test_checkpoint.py
"""
import importlib.util
import json
import pathlib
import sys
import types

_root = pathlib.Path(__file__).parent.resolve()
_pkg = types.ModuleType("h3ck")
_pkg.__path__ = [str(_root)]
sys.modules["h3ck"] = _pkg
_spec = importlib.util.spec_from_file_location("h3ck.checkpoint",
                                               _root / "checkpoint.py")
ck = importlib.util.module_from_spec(_spec)
sys.modules["h3ck.checkpoint"] = ck
_spec.loader.exec_module(ck)

fails = []


def check(label, got, want):
    if got != want:
        fails.append(label)
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


def ok(label, cond):
    check(label, bool(cond), True)


PLAN = {"chunks": [{"start": 0, "end": 141, "keep_from": 0, "pin": 0, "run": 141},
                   {"start": 102, "end": 243, "keep_from": 141, "pin": 39,
                    "run": 141}],
        "info": {"notes": ["some wording"]}, "total_frames": 243}


def variant(**changes):
    p = json.loads(json.dumps(PLAN))
    for k, v in changes.items():
        if k == "info":
            p["info"] = v
        else:
            p["chunks"][1][k] = v
    return p


print("the fingerprint follows the cuts, not the prose")
check("the same plan hashes the same",
      ck.plan_fingerprint(PLAN), ck.plan_fingerprint(json.loads(json.dumps(PLAN))))
check("re-wording info does not invalidate a checkpoint",
      ck.plan_fingerprint(variant(info={"notes": ["quite different wording"]})),
      ck.plan_fingerprint(PLAN))
check("a different total_frames alone does not either",
      ck.plan_fingerprint({**PLAN, "total_frames": 999}),
      ck.plan_fingerprint(PLAN))

print("but any field that moves a frame does")
for field, value in (("start", 110), ("end", 250), ("keep_from", 150), ("pin", 56)):
    ok(f"{field} changing invalidates it",
       ck.plan_fingerprint(variant(**{field: value})) != ck.plan_fingerprint(PLAN))

print("degenerate plans do not explode")
ok("no plan at all", isinstance(ck.plan_fingerprint(None), str))
ok("no chunks", isinstance(ck.plan_fingerprint({"chunks": []}), str))
check("and an empty plan is stable",
      ck.plan_fingerprint({"chunks": []}), ck.plan_fingerprint({}))

print("take folders are named safely")
ok("a path separator cannot escape the folder",
   "/" not in pathlib.Path(ck.take_dir("../../etc")).name and
   "\\\\" not in pathlib.Path(ck.take_dir("../../etc")).name)
ok("an empty name still lands somewhere",
   pathlib.Path(ck.take_dir("")).name == "take")

print()
print("NOT COVERED: saving and loading a real latent, which needs torch. The")
print("round trip is a torch.save/torch.load of a dict — the decision worth")
print("testing is whether a resume is ALLOWED, and that is the fingerprint.")
print()
print(f"{len(fails)} failure(s)" if fails else "checkpoint: all checks pass")
sys.exit(1 if fails else 0)
