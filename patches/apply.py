#!/usr/bin/env python3
"""Check, apply or revert this pack's ComfyUI core patches.

WHY IT EXISTS
  A ComfyUI or Manager update reverts a patch silently, and nothing errors
  afterwards -- the render just comes out wrong. Both of this pack's CORE
  patches have since been rewritten as subclasses inside the pack for exactly
  that reason, so what is left here is third-party.

    python3 patches/apply.py            # what is applied, what is not
    python3 patches/apply.py --apply    # apply whatever is missing
    python3 patches/apply.py --revert   # take them all back out

Idempotent: a patch already applied is reported and skipped, never applied
twice. Run it after every ComfyUI update.
"""

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def default_comfy():
    """The ComfyUI root, discovered rather than hardcoded.

    This pack normally lives at <ComfyUI>/custom_nodes/ComfyUI-H3-Toolkit, so
    three levels up from patches/ is the root. $COMFYUI_PATH wins when set, and
    --comfy overrides both. Same discovery order as test_windowing.py.
    """
    env = os.environ.get("COMFYUI_PATH")
    if env:
        return pathlib.Path(env)
    return HERE.parents[2]

# patch file -> the repo it applies inside, relative to the ComfyUI root
PATCHES = {
    # h3-modality-dim-context-windows.patch is SUPERSEDED -- windowing.py's
    # H3ContextHandler subclass carries it now. Harmless if still applied, but
    # nothing needs it.
    # h3-window-absolute-positions.patch is SUPERSEDED -- the offset lives in
    # video.py now. Applying it would double the offset.
    "depthanythingv2-contiguous.patch": "custom_nodes/ComfyUI-DepthAnythingV2",
    # h3-denoise-mask-velocity.patch is SHIPPED and the entry is GONE.
    # Comfy-Org/ComfyUI#15988 landed in 0.35: ldm/minimax/model.py:593-595 now
    # carries `out[0] = out[0] * denoise_mask` verbatim. The registry said to
    # remove this the moment it shipped, because applying it on top would scale
    # the velocity TWICE -- and apply.py had started reporting CONFLICT, which
    # is the same fact arriving as a complaint instead of a decision.
    # The .patch file stays for the record; nothing should apply it.
    # Exposes BICUBIC and HIGHBITRATE_* on the RTX VSR node. Purely additive to
    # a dropdown -- nothing renders differently until you pick a new level --
    # so a silent revert costs you options, not a wrong result.
    "nvidia_rtx_nodes_quality_levels.patch": "custom_nodes/comfyui_nvidia_rtx_nodes",
}


def git(cwd, *args, check=False):
    return subprocess.run(["git", "-C", str(cwd), *args], check=check,
                          capture_output=True, text=True)


def posix_patch(cwd, *args, patch=None):
    """`patch` instead of `git apply`, for a target that is not a repository.

    The Manager installs some packs as a bare folder, and `git apply` outside a
    work tree does not error -- it prints "Skipped patch" and exits 0. So a
    revert REPORTED SUCCESS and changed nothing, which is worse than failing.
    `patch --dry-run` detects applied/reverted correctly either way, and
    --forward stops it asking a question nobody is there to answer.
    """
    return subprocess.run(["patch", "-p1", "-d", str(cwd), "--forward", *args],
                          stdin=open(patch, "rb"), capture_output=True, text=True)


def is_repo(target):
    return (target / ".git").exists() or (target / ".git").is_file()


def state(root, patch, rel):
    """-> 'applied' | 'missing' | 'conflict' | reason it could not be read."""
    target = root / rel
    if not target.is_dir():
        return f"no such directory: {rel}"
    p = str(HERE / patch)
    if is_repo(target):
        if git(target, "apply", "--check", "--reverse", p).returncode == 0:
            return "applied"
        if git(target, "apply", "--check", p).returncode == 0:
            return "missing"
        return "conflict"
    # not a repo: `patch` can still tell applied from missing
    if posix_patch(target, "--dry-run", "-R", patch=p).returncode == 0:
        return "applied"
    if posix_patch(target, "--dry-run", patch=p).returncode == 0:
        return "missing"
    return "conflict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", type=pathlib.Path, default=default_comfy(),
                    help="the ComfyUI root; defaults to $COMFYUI_PATH, else "
                         "three levels up from this file")
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
            r = (git(target, "apply", "--reverse", str(HERE / patch))
                 if is_repo(target)
                 else posix_patch(target, "-R", patch=str(HERE / patch)))
            st = "REVERTED" if r.returncode == 0 else f"revert failed: {r.stderr.strip()}"
        elif st == "missing" and a.apply:
            r = (git(target, "apply", str(HERE / patch)) if is_repo(target)
                 else posix_patch(target, patch=str(HERE / patch)))
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
