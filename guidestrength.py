"""How hard the model has to obey its visual conditioning.

THE DIAL NOBODY EXPOSED
  H3 mixes noise into every visual condition row before it enters the sequence,
  and the amount is read from the conditioning:

      # comfy/ldm/minimax/model.py:502
      aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)  # 0.999
      if aug < 1.0:
          r = aug * r + (1.0 - aug) * noise

  Nothing in ComfyUI ever sets that key -- not core, not this pack -- so every
  render anyone has ever done ran at the 0.999 default. The guide rows arrive
  99.9% clean, and the model is told so a second time through their timestep
  label:

      # model.py:584
      seg_t = {..., "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug)}

  Both effects say the same thing: this row is finished, trust it. Which is
  exactly right for a single reference image, and exactly wrong for a guide clip
  that covers the whole timeline -- there the model has been handed a complete,
  authoritative specification of every output frame and it reproduces it. That is
  the "style shift did nothing" failure: not a weak style signal, an unarguable
  guide.

  Lower the value and the guide binds proportionally less: noisier rows, and a
  timestep that says so. 1.0 skips the noise branch entirely and is the hardest
  setting available.

ONE DIAL FOR BOTH KINDS
  `_cond_video_rows` walks `cond_video_latents`, which is keyframes AND references
  concatenated, and `seg_t` sets "cond" and "ref_img" from the same number. So
  this cannot weaken a guide clip while leaving reference images alone. On a graph
  with both, it moves both. On a guide-only graph (no refs) it is clean.

UNPROVEN. The mechanism is read straight from core and the arithmetic is plain,
but no render has been measured at any value other than the 0.999 default.
"""

import logging

import node_helpers

CATEGORY = "MiniMax H3/conditioning"

# core's defaults, so "leave it alone" is expressible and visible
VISUAL_DEFAULT = 0.999
AUDIO_DEFAULT = 0.999


class H3GuideStrength:
    """Set how strongly visual and audio conditioning rows bind."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "visual": ("FLOAT", {"default": VISUAL_DEFAULT, "min": 0.0, "max": 1.0,
                       "step": 0.001, "round": 0.001,
                       "tooltip": "Keyframes, guide clips and reference IMAGES. "
                                  "0.999 is core's default and what every render "
                                  "before this node used. 1.0 is harder still "
                                  "(no noise at all). Lower to let the prompt and "
                                  "LoRAs win against a dense guide -- try 0.9, "
                                  "then 0.7."}),
            "audio": ("FLOAT", {"default": AUDIO_DEFAULT, "min": 0.0, "max": 1.0,
                      "step": 0.001, "round": 0.001,
                      "tooltip": "Reference and keyframe AUDIO. Leave at 0.999 "
                                 "unless you are deliberately loosening a "
                                 "reference voice."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("How hard the model must obey its visual conditioning. Lower "
                   "the visual value to loosen a guide clip or a reference image; "
                   "0.999 reproduces the behaviour of every render before this "
                   "node existed.")

    def go(self, conditioning, visual, audio):
        visual, audio = float(visual), float(audio)
        notes = []
        if visual >= 1.0:
            notes.append("visual 1.0 — no noise mixed in at all, the hardest "
                         "setting; a full-length guide clip at this value fully "
                         "specifies the output")
        elif abs(visual - VISUAL_DEFAULT) < 1e-9:
            notes.append(f"visual {visual:g} — core's default, unchanged")
        else:
            notes.append(f"visual {visual:g} — guide and reference rows carry "
                         f"{100.0 * (1.0 - visual):.1f}% noise and are labelled "
                         f"at timestep {visual:g}")
        if abs(audio - AUDIO_DEFAULT) > 1e-9:
            notes.append(f"audio {audio:g}")

        out = node_helpers.conditioning_set_values(
            conditioning, {"minimax_visual_cond_noise_aug": visual,
                           "minimax_audio_cond_noise_aug": audio})
        info = "H3 GUIDE STRENGTH: " + "; ".join(notes)
        logging.info(info)
        return (out, info)


NODE_CLASS_MAPPINGS = {"H3GuideStrength": H3GuideStrength}
NODE_DISPLAY_NAME_MAPPINGS = {"H3GuideStrength": "H3 Guide Strength"}
