#!/usr/bin/env python3
"""Check, apply or revert this pack's ComfyUI core patches.

WHY IT EXISTS
  Three core files are patched, and a ComfyUI update reverts all of them
  silently. Nothing errors afterwards -- H3 context windows simply go back to
  windowing the wrong axis and placing every window at the clip origin, which
  looks like a flicker rather than a missing patch.

    python3 patches/apply.py            # what is applied, what is not
    python3 patches/apply.py --apply    # apply whatever is missing
    python3 patches/apply.py --revert   # take them all back out

Idempotent: a patch already applied is reported and skipped, never applied
twice. Run it after every ComfyUI update.
"""

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_COMFY = pathlib.Path("/mnt/c/SD/ComfyUI/Comfy-03-15-2026/ComfyUI")

# patch file -> the repo it applies inside, relative to the ComfyUI root
PATCHES = {
    "h3-modality-dim-context-windows.patch": ".",
    # h3-window-absolute-positions.patch is SUPERSEDED -- the offset lives in
    # video.py now. Applying it would double the offset.
    "depthanythingv2-contiguous.patch": "custom_nodes/ComfyUI-DepthAnythingV2",
}


def git(cwd, *args, check=False):
    return subprocess.run(["git", "-C", str(cwd), *args], check=check,
                          capture_output=True, text=True)


def state(root, patch, rel):
    """-> 'applied' | 'missing' | 'conflict' | reason it could not be read."""
    target = root / rel
    if not (target / ".git").exists() and not (target / ".git").is_file():
        return f"no git repo at {rel}"
    p = str(HERE / patch)
    if git(target, "apply", "--check", "--reverse", p).returncode == 0:
        return "applied"
    if git(target, "apply", "--check", p).returncode == 0:
        return "missing"
    return "conflict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", type=pathlib.Path, default=DEFAULT_COMFY)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if not a.comfy.exists():
        print(f"no ComfyUI at {a.comfy} — pass --comfy")
        return 2

    bad = 0
    for patch, rel in PATCHES.items():
        st = state(a.comfy, patch, rel)
        target = a.comfy / rel
        if st == "applied" and a.revert:
            r = git(target, "apply", "--reverse", str(HERE / patch))
            st = "REVERTED" if r.returncode == 0 else f"revert failed: {r.stderr.strip()}"
        elif st == "missing" and a.apply:
            r = git(target, "apply", str(HERE / patch))
            st = "APPLIED" if r.returncode == 0 else f"apply failed: {r.stderr.strip()}"
        elif st == "conflict":
            # neither direction is clean: core moved under the patch
            bad += 1
            st = ("CONFLICT — core has changed here. The patch needs rebasing; "
                  "see patches/README.md for what it does and why.")
        print(f"  {st:<12} {patch}")
    if not (a.apply or a.revert):
        print("\n  --apply to apply what is missing, --revert to remove them")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
