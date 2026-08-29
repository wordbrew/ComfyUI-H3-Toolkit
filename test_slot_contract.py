"""Every node's inputs and outputs must stay in the order saved graphs expect.

WHY THIS FILE EXISTS
  ComfyUI stores widget values POSITIONALLY, in `widgets_values`, and links by
  slot INDEX. So inserting an input anywhere but the end silently reassigns every
  value after it in every saved workflow, and inserting an output silently
  rewires every link after it. Nothing errors. The node comes up reading a
  neighbour's number and rendering something plausible.

  It is written in CLAUDE.md and it has still been broken four times in three
  days -- H3ChunkPlan's `context`, H3ContextWindows' `absolute_window_positions`
  and again its `split_conds_to_windows`, each caught only by the symptom. This
  pins the orders that saved workflows already depend on, so the next one fails
  here instead of in a render.

  ADDING an entry to the END of a list below is correct and expected. Changing
  or reordering an existing one means saved graphs break: if that is genuinely
  intended, the workflows in workflows/ have to be migrated in the same commit.

    python3 test_slot_contract.py
"""

import sys
import types

# longform.py reaches torch.nn.functional through avlatent, so the stub needs the
# shape of a package. Only names touched at IMPORT time matter -- nothing here
# calls a tensor op.
_torch = types.ModuleType("torch")
_torch.__path__ = []
_torch.Tensor = type("Tensor", (), {})
_nn = types.ModuleType("torch.nn")
_nn.__path__ = []
_fn = types.ModuleType("torch.nn.functional")
_nn.functional = _fn
_torch.nn = _nn
sys.modules["torch"] = _torch
sys.modules["torch.nn"] = _nn
sys.modules["torch.nn.functional"] = _fn

fails = []

# name -> (module, class, inputs in declared order, outputs in declared order)
CONTRACT = {
    "H3ContextWindows": (
        "windowing",
        ["model", "window_frames", "overlap_frames", "schedule", "fuse_method",
         "freenoise", "causal_window_fix", "absolute_window_positions",
         "split_conds_to_windows"],
        ["model", "info"]),
    "H3WindowPlan": (
        "windowing",
        ["window_frames", "overlap_frames", "mode", "total_frames", "windows"],
        ["length", "window_count", "info"]),
    "H3ChunkPlan": (
        "longform",
        ["chunk_frames", "chunk_mode", "source_images", "total_frames",
         "scene_threshold", "min_chunk", "render_width", "render_height",
         "ref_tokens", "context"],
        ["plan", "chunk_count", "info"]),
    "H3LongFormLinks": (
        "prompt_links",
        ["head", "beats", "tail", "link_index", "seconds_per_link", "seed",
         "subject_def_1", "retention_1", "expected_count", "task_type",
         "chunk_frames", "soundscape", "music", "chunk_plan"],
        ["prompt", "length", "seed", "link_count", "plan", "lint", "clause"]),
    "H3RefBudget": (
        "budget", None,
        ["info", "ref_share", "seq_len", "ref_tokens"]),
    "H3ChunkOpen": (
        "chunkrun", None,
        ["images", "mask", "audio", "length", "chunk_index", "flow", "info",
         "chunk_count", "keyframe", "context", "extra", "pin", "prev_latent"]),
    "H3ChunkSlice": (
        "chunkrun", None,
        ["images", "mask", "audio", "length", "chunk_index", "flow", "info",
         "chunk_count", "keyframe", "context", "extra", "pin", "prev_latent"]),
    "H3ChunkClose": ("chunkrun", None, ["images", "info", "audio"]),
    "H3ChunkContext": (
        "chunkrun",
        ["images", "mask", "pin", "context_images"],
        ["images", "mask", "pinned", "info"]),
    "H3SwapPrompt": (
        "video",
        ["conditioning", "clip", "prompt"],
        ["conditioning", "info"]),
    "H3ChunkLatentContext": (
        "chunkrun",
        ["latent", "source_latent", "context_length", "audio_feather_ticks"],
        ["latent", "trim_frames", "info"]),
}


def declared_inputs(cls):
    """Names in the order ComfyUI builds them: required, then optional."""
    spec = cls.INPUT_TYPES()
    out = list(spec.get("required") or {})
    out += list(spec.get("optional") or {})
    return out


# Some modules use relative imports, so load the pack AS a package rather than
# importing each file flat. Importing __init__ registers the routes and would
# pull in comfy, so build the package shell and import the submodules under it.
import importlib.util                                          # noqa: E402
import pathlib as _pathlib                                     # noqa: E402

_ROOT = _pathlib.Path(__file__).resolve().parent
_pkg = types.ModuleType("h3slots")
_pkg.__path__ = [str(_ROOT)]
_pkg.__package__ = "h3slots"
sys.modules["h3slots"] = _pkg


def load(module):
    spec = importlib.util.spec_from_file_location(f"h3slots.{module}",
                                                  _ROOT / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"h3slots.{module}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Modules that reach into comfy at import time cannot be checked outside a
# ComfyUI install. Skip them and SAY SO rather than failing: run this from
# inside the install to cover them too. Everything torch-free is checked here.
skipped = []

for name, (module, want_in, want_out) in CONTRACT.items():
    try:
        mod = load(module)
        cls = mod.NODE_CLASS_MAPPINGS[name]
    except ModuleNotFoundError as exc:
        if "comfy" in str(exc) or "node_helpers" in str(exc):
            skipped.append(f"{name} ({module}: {exc})")
            continue
        fails.append(f"{name}: could not load from {module}: "
                     f"{type(exc).__name__}: {exc}")
        continue
    except Exception as exc:
        fails.append(f"{name}: could not load from {module}: "
                     f"{type(exc).__name__}: {exc}")
        continue

    if want_in is not None:
        got = declared_inputs(cls)
        if got != want_in:
            extra = got[len(want_in):] if got[:len(want_in)] == want_in else None
            if extra:
                fails.append(f"{name} inputs: {extra} appended — correct, but "
                             f"add them to CONTRACT in this file")
            else:
                fails.append(f"{name} inputs REORDERED, which breaks every saved "
                             f"workflow:\n      was  {want_in}\n      now  {got}")

    got = list(getattr(cls, "RETURN_NAMES", ()) or [])
    if got != want_out:
        extra = got[len(want_out):] if got[:len(want_out)] == want_out else None
        if extra:
            fails.append(f"{name} outputs: {extra} appended — correct, but add "
                         f"them to CONTRACT in this file")
        else:
            fails.append(f"{name} outputs REORDERED, which silently rewires every "
                         f"saved link:\n      was  {want_out}\n      now  {got}")

    types_ = getattr(cls, "RETURN_TYPES", ())
    if len(types_) != len(got):
        fails.append(f"{name}: {len(types_)} RETURN_TYPES against "
                     f"{len(got)} RETURN_NAMES")

if fails:
    print("FAIL — the slot contract is broken")
    for f in fails:
        print("  " + f)
    sys.exit(1)
checked = len(CONTRACT) - len(skipped)
print(f"slot contract: {checked} node(s) hold their order")
for s_ in skipped:
    print(f"  SKIPPED {s_}")
if skipped:
    print("  (run this from inside a ComfyUI install to cover those)")
