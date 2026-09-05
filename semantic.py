"""Blend a semantic adapter into H3's conditioning — over the PROMPT only.

WHAT THE ADAPTER IS
  A small MLP, 5120 -> 512 -> 512 -> 5120 with SiLU, distilled from a
  cross-architecture bridge (SenseNova U1.5 as semantic teacher, H3 as student).
  It reads a conditioning token and returns a "more semantically structured"
  version of it, and the node blends the two:

      hybrid = h + alpha * (projected - h)

  It aims at the things H3 recognises but mis-relates: which hand holds what,
  left/right ordering, transparent versus reflective, and whether an action was
  explicitly NOT requested. Weights are the author's; this is our own loader and
  our own application, because the difference below is the entire point.

WHY WE DO NOT APPLY IT EVERYWHERE
  The published node blends every token in the conditioning. That is correct for
  fl2va, where the conditioning is text and nothing else, and its author says
  plainly that ref2va is unsupported and that reference AUDIO degraded lip-sync
  when it was included.

  The reason is distributional. In ref2va the conditioning is a packed sequence:

      <Picture 1>: [vision] <Picture 2>: [vision] <Audio 1>: <the prompt>

  Those vision and audio-label tokens do not exist in fl2va, so the adapter never
  saw them in training and is being asked to transform a distribution it has no
  map for. A LoRA crosses between the two paths because it acts on the model's
  COMPUTATION, below where the conditioning differs. An adapter acts on the
  REPRESENTATION, which is exactly what differs.

  So restrict it to the span it was trained on. The prompt is a contiguous
  SUFFIX -- `comfy/text_encoders/minimax.py` emits every reference item first and
  calls `add_text(prompt)` last -- so masking the trailing N tokens leaves every
  reference block untouched, byte for byte.

UNPROVEN. The adapter's own evidence is representation fidelity (cosine 0.749 on
a held-out 160-prompt set) plus two qualitative A/B clips. Nothing measures video
quality, and nothing at all has been measured on ref2va. `apply_to` defaults to
the prompt span; `everything` reproduces the published behaviour for comparison.
"""

import logging
import os

import torch

CATEGORY = "MiniMax H3/prompt"

_CACHE = {}


def _folder():
    """`models/semantic_bridge`, the layout the adapter is published for."""
    try:
        import folder_paths
        root = folder_paths.models_dir
    except Exception:  # pragma: no cover - outside ComfyUI
        return None
    return os.path.join(root, "semantic_bridge")


def _adapters():
    d = _folder()
    if not d or not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))


class _Student(torch.nn.Module):
    """The published adapter's shape. Plain fc1/fc2/fc3 with SiLU between."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5120, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 5120)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


def _load(name, device):
    key = (name, str(device))
    if key in _CACHE:
        return _CACHE[key]
    import safetensors.torch
    path = os.path.join(_folder() or "", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"H3 Semantic Bridge: {name} is not in models/semantic_bridge. "
            f"Put the adapter .safetensors there.")
    w = safetensors.torch.load_file(path, device="cpu")
    want = {"fc1.weight": (512, 5120), "fc2.weight": (512, 512),
            "fc3.weight": (5120, 512)}
    for k, shape in want.items():
        if k not in w:
            raise ValueError(f"H3 Semantic Bridge: {name} has no {k}. This does "
                             f"not look like a semantic-bridge adapter.")
        if tuple(w[k].shape) != shape:
            raise ValueError(f"H3 Semantic Bridge: {k} is {tuple(w[k].shape)}, "
                             f"expected {shape}.")
    m = _Student()
    with torch.no_grad():
        for k, v in w.items():
            if hasattr(m, k.split(".")[0]):
                getattr(m, k.split(".")[0]).__getattr__(k.split(".")[1]).copy_(
                    v.float())
    m = m.to(device=device, dtype=torch.float32).eval()
    _CACHE[key] = m
    return m


def span(n_text, total):
    """(start_index, note) for the prompt suffix, or (None, why not).

    Torch-free on purpose: this is the decision the node exists to make, and it
    is the part worth testing without ComfyUI. `None` means fall back to the
    whole sequence and SAY SO -- masking nothing while claiming to protect
    references is the failure this function exists to make impossible.
    """
    n_text, total = int(n_text), int(total)
    if n_text <= 0:
        return None, f"span {n_text} of {total} — not a usable suffix"
    if n_text >= total:
        return None, (f"span {n_text} of {total} — not a usable suffix, the "
                      f"prompt fills the sequence and there is nothing to "
                      f"protect")
    lo = total - n_text
    return lo, (f"prompt span = last {n_text} of {total} tokens "
                f"({100.0 * n_text / total:.0f}%); {lo} reference token(s) "
                f"untouched")


def _rms(x):
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


def prompt_token_count(clip, prompt):
    """How many conditioning tokens the PROMPT occupies, at the end.

    Tokenizing the prompt with no reference items runs the same `add_text` path
    the full call ends with, so the count is exact. Vision blocks expand to more
    embeddings than entries, which is why this is measured on the prompt alone
    and applied as a suffix rather than counted forward from the start.
    """
    toks = clip.tokenize(prompt)
    if isinstance(toks, dict):
        for v in toks.values():
            if isinstance(v, list) and v and isinstance(v[0], list):
                return len(v[0])
    return 0


class H3SemanticBridge:
    """Blend a semantic adapter into the prompt span of H3 conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "clip": ("CLIP", {"tooltip": "The same CLIP the conditioning was "
                              "built with — used only to count how many tokens "
                              "the prompt occupies."}),
            "prompt": ("STRING", {"multiline": True, "default": "",
                       "tooltip": "The SAME prompt string the reference node "
                                  "was given. If it differs, the span is wrong "
                                  "and the wrong tokens get transformed."}),
            "adapter": ((_adapters() or ["(none in models/semantic_bridge)"]),),
            "alpha": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0,
                      "step": 0.01,
                      "tooltip": "Blend strength. 0.10 is the author's "
                                 "recommendation; 0.15 is what the published "
                                 "A/B examples used to make the difference "
                                 "visible. 0 is a no-op."}),
            "magnitude_match": (["per_token", "global", "none"],
                                {"default": "per_token"}),
            "apply_to": (["prompt only", "everything"],
                         {"default": "prompt only",
                          "tooltip": "`prompt only` masks the blend to the "
                                     "trailing prompt tokens, leaving every "
                                     "reference block untouched — the whole "
                                     "reason this node exists. `everything` "
                                     "reproduces the published behaviour, for "
                                     "comparison on fl2va."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Blend a distilled semantic adapter into H3 conditioning, "
                   "over the prompt tokens only so reference blocks are left "
                   "alone.")

    def go(self, conditioning, clip, prompt, adapter, alpha, magnitude_match,
           apply_to):
        if float(alpha) == 0.0:
            return (conditioning, "alpha 0 — passthrough, nothing applied")

        n_text = prompt_token_count(clip, prompt) if apply_to == "prompt only" else 0
        out, notes = [], []
        for entry in conditioning:
            h = entry[0].float()
            if h.ndim != 3 or h.shape[-1] != 5120:
                raise ValueError(
                    f"H3 Semantic Bridge: expected H3 conditioning "
                    f"[B, T, 5120], got {tuple(entry[0].shape)}.")
            T = int(h.shape[1])
            model = _load(adapter, h.device)
            with torch.inference_mode():
                projected = model(_rms(h))
            if magnitude_match == "per_token":
                a = torch.sqrt(projected.pow(2).mean(-1, keepdim=True) + 1e-8)
                b = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-8)
                projected = projected * (b / a)
            elif magnitude_match == "global":
                a = torch.sqrt(projected.pow(2).mean() + 1e-8)
                b = torch.sqrt(h.pow(2).mean() + 1e-8)
                projected = projected * (b / a)

            gain = float(alpha)
            if apply_to == "prompt only":
                lo, note = span(n_text, T)
                notes.append(note if lo is not None
                             else note + ", applied to everything")
                if lo is not None:
                    m = torch.zeros(1, T, 1, device=h.device, dtype=h.dtype)
                    m[:, lo:, :] = 1.0
                    gain = float(alpha) * m
            else:
                notes.append(f"applied to all {T} tokens")

            hybrid = (h + gain * (projected - h)).to(entry[0].dtype)
            meta = dict(entry[1])
            meta["h3_semantic_bridge"] = {"adapter": adapter,
                                          "alpha": float(alpha),
                                          "magnitude_match": magnitude_match,
                                          "apply_to": apply_to}
            out.append([hybrid, meta])

        info = (f"H3 SEMANTIC BRIDGE: {adapter} at alpha {float(alpha):.2f}, "
                f"{magnitude_match}\n  " + "\n  ".join(notes or ["(no entries)"]))
        logging.info(info.replace("\n", " | "))
        return (out, info)


NODE_CLASS_MAPPINGS = {"H3SemanticBridge": H3SemanticBridge}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SemanticBridge": "H3 Semantic Bridge (prompt span)"}
