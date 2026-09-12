"""Stop a chained take between chunks, look at it, and carry on.

WHY A LOOP CANNOT SIMPLY PAUSE
  `H3ChunkOpen` expands the whole take into one graph and `H3ChunkClose` hands
  each chunk's sampler output to the next as `prev_latent`. It is one queue item
  from first frame to last, so there is no point at which ComfyUI could stop and
  wait for you. Gating means ENDING the graph and starting another one, which
  means whatever the next chunk needs has to survive on disk.

WHAT HAS TO SURVIVE, AND IN WHICH FORM
  The carry is a LATENT, and it has to stay one. H3ChunkClose's own note says
  each chunk's latent reaches the next one's `prev_latent` "with no VAE round
  trip", because that round trip is what climbed local contrast +12%/+7%/+5%
  down a chain (measured 2026-08-29). A checkpoint that decoded to pixels and
  re-encoded on resume would add a generation of photocopy at every gate --
  turning a review tool into the thing it is meant to catch.

  So: the latent, the audio, the chunk index, and a fingerprint of the plan.
  Decoding to LOOK at a chunk is free; decoding to CARRY it is not.

THE PLAN FINGERPRINT
  Resuming against an edited plan is the failure worth refusing rather than
  reporting. If the cuts moved, chunk 3's prefix no longer belongs to chunk 2's
  tail and the seam it produces is nobody's fault but this node's.
"""

import hashlib
import json
import logging
import os

CATEGORY = "MiniMax H3/long-form"


def _root():
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:                      # pragma: no cover - outside ComfyUI
        base = os.path.join(os.path.dirname(__file__), "_checkpoints")
    d = os.path.join(base, "h3_takes")
    os.makedirs(d, exist_ok=True)
    return d


def plan_fingerprint(plan):
    """A short hash of the cuts, so a resume can tell the plan changed.

    Only the fields that decide where a chunk STARTS and how much of it is
    carried -- not the info text, which changes wording without changing a frame.
    """
    chunks = (plan or {}).get("chunks") or []
    spine = [[int(c.get("start", 0)), int(c.get("end", 0)),
              int(c.get("keep_from", 0)), int(c.get("pin", 0))] for c in chunks]
    return hashlib.sha1(json.dumps(spine).encode()).hexdigest()[:12]


def take_dir(take):
    safe = "".join(ch for ch in (take or "take") if ch.isalnum() or ch in "-_ ").strip()
    d = os.path.join(_root(), safe or "take")
    os.makedirs(d, exist_ok=True)
    return d


class H3ChunkCheckpoint:
    """Save a chunk's carry so the take can be resumed after you have seen it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "This chunk's SAMPLER output — the "
                                  "same latent H3 Chunk Close would hand the "
                                  "next chunk. Not a decoded image: the VAE "
                                  "round trip is what climbs contrast down a "
                                  "chain."}),
            "chunk_index": ("INT", {"forceInput": True,
                            "tooltip": "From H3 Chunk Open. Without it every "
                                       "chunk would overwrite chunk 0."}),
            "take": ("STRING", {"default": "take", "multiline": False,
                     "tooltip": "A folder name under output/h3_takes."}),
        }, "optional": {
            "plan": ("H3_CHUNK_PLAN", {"tooltip": "Recorded as a fingerprint so "
                     "a resume against an edited plan is refused rather than "
                     "producing a seam nobody can explain."}),
            "audio": ("AUDIO",),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Write a chunk's carry to disk so the take can be reviewed "
                   "and resumed. Passes the latent straight through.")

    def go(self, latent, chunk_index, take, plan=None, audio=None, enabled=True):
        if not enabled:
            return (latent, "H3 CHECKPOINT: disabled — nothing written")
        import torch
        d = take_dir(take)
        ci = int(chunk_index)
        path = os.path.join(d, f"chunk_{ci:03d}.pt")
        payload = {"chunk_index": ci, "latent": latent,
                   "plan_fingerprint": plan_fingerprint(plan) if plan else "",
                   "audio": audio}
        torch.save(payload, path)
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"take": take, "last_chunk": ci,
                       "plan_fingerprint": payload["plan_fingerprint"]}, fh,
                      indent=2)
        info = (f"H3 CHECKPOINT: chunk {ci} saved to {os.path.basename(d)}/"
                f"{os.path.basename(path)}\n  resume the next run from chunk "
                f"{ci + 1}")
        logging.info(info.replace("\n", " | "))
        return (latent, info)


class H3ChunkResume:
    """Load a saved chunk's carry, to start a take part way through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "take": ("STRING", {"default": "take", "multiline": False}),
            "from_chunk": ("INT", {"default": 0, "min": 0, "max": 4096,
                           "tooltip": "The chunk to START at. Its carry comes "
                                      "from the checkpoint written by the chunk "
                                      "BEFORE it. 0 means begin from nothing."}),
        }, "optional": {
            "plan": ("H3_CHUNK_PLAN", {"tooltip": "Checked against the plan the "
                     "checkpoint was written under."}),
        }}

    RETURN_TYPES = ("LATENT", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("prev_latent", "audio", "from_chunk", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Load a checkpointed chunk's carry so a take can resume "
                   "without regenerating what you have already approved.")

    def go(self, take, from_chunk, plan=None):
        import torch
        d = take_dir(take)
        want = int(from_chunk) - 1
        if want < 0:
            return (None, None, 0, "H3 RESUME: starting from the beginning")
        path = os.path.join(d, f"chunk_{want:03d}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"H3 Resume: no checkpoint for chunk {want} in "
                f"{os.path.basename(d)}. Run up to that chunk with H3 Chunk "
                f"Checkpoint enabled before resuming past it.")
        data = torch.load(path, weights_only=False)
        # REFUSE, do not warn. A resume against moved cuts puts chunk N's prefix
        # against a tail that is no longer there, and the seam it makes looks
        # like the model's fault.
        if plan is not None and data.get("plan_fingerprint"):
            now = plan_fingerprint(plan)
            if now != data["plan_fingerprint"]:
                raise ValueError(
                    f"H3 Resume: the plan has changed since chunk {want} was "
                    f"saved ({data['plan_fingerprint']} -> {now}). The carry "
                    f"belongs to different cuts. Re-run from chunk 0, or resume "
                    f"with the plan it was made under.")
        info = (f"H3 RESUME: chunk {want}'s carry loaded — this run starts at "
                f"chunk {int(from_chunk)}")
        logging.info(info)
        return (data.get("latent"), data.get("audio"), int(from_chunk), info)


NODE_CLASS_MAPPINGS = {"H3ChunkCheckpoint": H3ChunkCheckpoint,
                       "H3ChunkResume": H3ChunkResume}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ChunkCheckpoint": "H3 Chunk Checkpoint",
                              "H3ChunkResume": "H3 Chunk Resume"}
