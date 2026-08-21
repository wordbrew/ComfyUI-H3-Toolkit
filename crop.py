"""Crop a clip to its subject, generate small, paste back.

The speed lever. Everything else we can change — steps, scheduler, quantisation,
sampler — is a percentage. This is the one that changes the order of magnitude,
because H3 denoises every token every step and a full-frame render spends most of
them on scenery that is pinned anyway.

Pair with H3 Mask Inpaint: crop -> H3 conditioning at the crop's size -> mask
inpaint -> sample -> decode -> uncrop. `crop_scale 0` turns the whole thing off
and passes full frames, so the same graph runs both ways.

The planning is in cropplan.py, which has the reasoning and is torch-free so it
can be tested. This file is the tensor work and the node surface.
"""

import logging

import torch
import torch.nn.functional as Fn

from .cropplan import plan

CATEGORY = "MiniMax H3/mask"


def _bboxes(mask, threshold=0.5):
    """Per-frame inclusive (x0, y0, x1, y1), or None where the mask is empty.

    Vectorised over frames — a Python loop calling nonzero() per frame is a real
    cost at 345 frames and there is no need for one.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    b = mask > float(threshold)                       # [N, H, W]
    n, h, w = b.shape
    rows = b.any(dim=2)                               # [N, H]
    cols = b.any(dim=1)                               # [N, W]
    hit = rows.any(dim=1)

    ar = torch.arange(h, device=b.device)
    aw = torch.arange(w, device=b.device)
    y0 = torch.where(rows, ar, torch.full_like(ar, h)).min(dim=1).values
    y1 = torch.where(rows, ar, torch.full_like(ar, -1)).max(dim=1).values
    x0 = torch.where(cols, aw, torch.full_like(aw, w)).min(dim=1).values
    x1 = torch.where(cols, aw, torch.full_like(aw, -1)).max(dim=1).values

    out = []
    for i in range(n):
        out.append(None if not bool(hit[i]) else
                   (int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i])))
    return out


def _uniform(boxes):
    """True when every frame shares one box — the static case, sliced in one go."""
    f = boxes[0]
    return all(b == f for b in boxes)


class H3SubjectCrop:
    """Cut the clip down to the subject, so the model only renders what changes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The source clip."}),
            "mask": ("MASK", {"tooltip": "Subject mask, one per frame. Only its "
                                         "extent is used — this decides WHERE to "
                                         "cut, not what to regenerate."}),
            "mode": (["static", "tracked"], {"default": "static",
                     "tooltip": "static: one box covering the subject's whole path "
                                "— nothing moves, nothing can wobble, and on a "
                                "locked-off camera it is nearly as tight as "
                                "tracking. tracked: constant size, position holds "
                                "still and moves only when the subject would leave "
                                "the box. Use tracked when the subject crosses the "
                                "frame."}),
            "crop_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0,
                           "step": 0.05,
                           "tooltip": "Box size as a multiple of the subject. 1.5 "
                                      "leaves half a subject-width of context. "
                                      "SET 0 TO DISABLE cropping entirely and pass "
                                      "full frames — the whole graph then runs "
                                      "uncropped with no rewiring."}),
        }, "optional": {
            "aspect_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0,
                             "step": 0.01,
                             "tooltip": "0 = let the subject decide. Otherwise "
                                        "width/height, e.g. 0.571 for 640x1120. "
                                        "Pin it if H3 turns out to dislike the "
                                        "shape the subject produces — both axes "
                                        "still have to land on the grid, so the "
                                        "ratio is approximate."}),
            "smooth_window": ("INT", {"default": 16, "min": 1, "max": 121,
                              "tooltip": "tracked only. Frames of median filtering "
                                         "that define SUSTAINED motion. Bigger "
                                         "ignores more mask noise; too big and the "
                                         "crop lags a real move."}),
            "divisible_by": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8,
                             "tooltip": "H3 needs 32. Leave it."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "H3_CROP", "STRING")
    RETURN_NAMES = ("images", "mask", "width", "height", "crop_data", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Crop a clip to its subject and report the size to render at. "
                   "Wire width/height into the H3 conditioning node and crop_data "
                   "into H3 Subject Uncrop. crop_scale 0 disables it.")

    def go(self, images, mask, mode, crop_scale, aspect_ratio=0.0,
           smooth_window=16, divisible_by=32):
        n, ih, iw = images.shape[0], images.shape[1], images.shape[2]
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        if m.shape[0] != n:
            raise ValueError(
                f"{n} frame(s) of video but {m.shape[0]} of mask. They have to "
                f"match — put H3 Match Source Clip in front of both.")
        if m.shape[-2:] != (ih, iw):
            raise ValueError(
                f"mask is {m.shape[-1]}x{m.shape[-2]} but the clip is {iw}x{ih}.")

        boxes, info = plan(_bboxes(m), iw, ih, mode=mode, crop_scale=crop_scale,
                           aspect_ratio=aspect_ratio, divisible_by=divisible_by,
                           smooth_window=smooth_window)
        w, h = boxes[0]["width"], boxes[0]["height"]

        if _uniform(boxes):
            b = boxes[0]
            out_i = images[:, b["y"]:b["y"] + h, b["x"]:b["x"] + w, :]
            out_m = m[:, b["y"]:b["y"] + h, b["x"]:b["x"] + w]
        else:
            out_i = torch.empty((n, h, w, images.shape[3]), dtype=images.dtype,
                                device=images.device)
            out_m = torch.empty((n, h, w), dtype=m.dtype, device=m.device)
            for i, b in enumerate(boxes):
                y, x = b["y"], b["x"]
                out_i[i] = images[i, y:y + h, x:x + w, :]
                out_m[i] = m[i, y:y + h, x:x + w]

        saved = 1.0 - (w * h) / float(iw * ih)
        text = (f"{info['mode']}: {iw}x{ih} -> {w}x{h} ({info.get('aspect', 0)}:1), "
                f"{saved * 100:.0f}% fewer pixels per frame")
        if info["mode"] != "disabled":
            text += f", {info['moves']} move(s)"
            if not info.get("covered", True):
                text += ("\nWARNING: the subject does not fit the box on every "
                         "frame and is clipped there. Raise crop_scale, or the "
                         "subject is simply larger than the box the image allows.")
        if info.get("note"):
            text += "\n" + info["note"]
        logging.info("H3SubjectCrop: %s", text.replace("\n", " "))

        crop_data = {"boxes": boxes, "image_width": iw, "image_height": ih,
                     "frames": n}
        return {"ui": {"h3char": [text]},
                "result": (out_i, out_m, int(w), int(h), crop_data, text)}


class H3SubjectUncrop:
    """Put the rendered crop back into the full frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The rendered crop."}),
            "source_images": ("IMAGE", {"tooltip": "The full frames the crop came "
                                                   "from — same ones you fed "
                                                   "H3 Subject Crop."}),
            "crop_data": ("H3_CROP",),
            "feather": ("INT", {"default": 16, "min": 0, "max": 256,
                        "tooltip": "Pixels of blend at the crop border. The model "
                                   "cannot see past the crop, so its edge pixels "
                                   "have less context than its middle — a soft "
                                   "border hides that. 0 butts them together."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Paste a rendered crop back into the original frames with a "
                   "feathered border. Everything outside the crop is the source, "
                   "untouched and unresampled.")

    def go(self, images, source_images, crop_data, feather):
        boxes = crop_data["boxes"]
        n = min(len(boxes), source_images.shape[0], images.shape[0])
        out = source_images.clone()
        h, w = boxes[0]["height"], boxes[0]["width"]

        gen = images
        if gen.shape[1] != h or gen.shape[2] != w:
            logging.warning(
                "H3SubjectUncrop: render is %dx%d but the crop was %dx%d — "
                "rescaling to fit. The H3 conditioning node's width/height should "
                "be wired from H3 Subject Crop so this never happens.",
                gen.shape[2], gen.shape[1], w, h)
            gen = Fn.interpolate(gen.movedim(-1, 1), size=(h, w), mode="bicubic",
                                 align_corners=False,
                                 antialias=True).clamp(0, 1).movedim(1, -1)

        iw = crop_data.get("image_width", source_images.shape[2])
        ih = crop_data.get("image_height", source_images.shape[1])
        cache = {}
        for i in range(n):
            b = boxes[i]
            y, x = b["y"], b["x"]
            # which sides actually have a seam to hide
            key = (x > 0, y > 0, x + w < iw, y + h < ih)
            if key not in cache:
                cache[key] = self._alpha(h, w, feather, key,
                                         images.device, out.dtype).unsqueeze(-1)
            a = cache[key]
            region = out[i, y:y + h, x:x + w, :]
            out[i, y:y + h, x:x + w, :] = (gen[i].to(out.dtype) * a
                                           + region * (1.0 - a))
        return (out,)

    @staticmethod
    def _alpha(h, w, feather, sides, device, dtype):
        """1 in the middle, ramping to 0 only on sides that have a seam.

        `sides` is (left, top, right, bottom): True where the crop box is INSIDE
        the frame and there is a real boundary between generated and kept pixels.

        Feathering a side that is flush with the image edge is not a no-op, it is
        a defect: there is nothing to blend into, so the ramp mixes VAE-decoded
        output with raw source over `feather` pixels. Those two differ in
        sharpness and level, so the result is a soft brighter band ringing the
        picture — which is exactly what a full-frame crop produced before this.
        """
        f = int(feather)
        if f <= 0 or not any(sides):
            return torch.ones((h, w), device=device, dtype=dtype)
        f = min(f, max(1, min(h, w) // 2))
        ramp = (torch.arange(f, device=device, dtype=dtype) + 1.0) / (f + 1.0)
        left, top, right, bottom = sides
        ay = torch.ones(h, device=device, dtype=dtype)
        ax = torch.ones(w, device=device, dtype=dtype)
        if top:
            ay[:f] = ramp
        if bottom:
            ay[-f:] = ramp.flip(0)
        if left:
            ax[:f] = ramp
        if right:
            ax[-f:] = ramp.flip(0)
        return torch.minimum(ay.unsqueeze(1), ax.unsqueeze(0))


class H3ApplyCrop:
    """Cut a DERIVED stream with the same boxes the subject crop used.

    Anything computed from the source frames — a depth pass, pose, canny, a
    second copy at another scale — has to travel the same path or it stops
    describing the frames it is paired with.

    It will not error if you skip this: a reference video gets its own latent
    block at its own size, so a full-frame depth pass against a cropped target
    runs happily. It is simply wrong. The whole value of a depth reference is
    spatial correspondence with what is being rendered, and cropping one side of
    that pairing and not the other is precisely what breaks it.

    Streams computed at a different scale to the source are handled: the boxes
    are scaled to match, so a half-resolution depth pass crops correctly.

    THE OTHER WAY ROUND IS OFTEN BETTER. Putting H3 Subject Crop BEFORE the depth
    model means depth is computed on the crop and cannot be misaligned at all —
    and it is cheaper, because the depth model sees fewer pixels. The difference
    is that estimators normalise relative depth per image, so depth-of-a-crop is
    not the same as a-crop-of-depth: the range spreads across the subject instead
    of the whole room. That is usually an improvement. Use this node when you want
    the full-scene normalisation preserved, or when the stream cannot be recomputed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "A stream derived from the same source "
                                            "frames — depth, pose, canny, a rescaled "
                                            "copy."}),
            "crop_data": ("H3_CROP", {"tooltip": "From H3 Subject Crop."}),
        }, "optional": {
            "mask": ("MASK",),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "mask", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Apply an existing subject crop to another stream, so a depth or "
                   "pose pass keeps describing the frames it is paired with.")

    def go(self, images, crop_data, mask=None):
        boxes = crop_data["boxes"]
        iw, ih = crop_data["image_width"], crop_data["image_height"]
        n, h, w = images.shape[0], images.shape[1], images.shape[2]

        if len(boxes) < n:
            raise ValueError(
                f"the crop plan covers {len(boxes)} frame(s) but this stream has "
                f"{n}. Both have to come from the same clip.")

        sx, sy = w / float(iw), h / float(ih)
        if abs(sx - sy) > 0.02:
            raise ValueError(
                f"this stream is {w}x{h} against the cropped clip's {iw}x{ih} — the "
                f"aspect differs, so it is not a rescaled copy of the same frames "
                f"and the boxes cannot be mapped onto it.")

        def sc(v, s, limit):
            return max(0, min(int(round(v * s)), limit))

        bw = sc(boxes[0]["width"], sx, w) or 1
        bh = sc(boxes[0]["height"], sy, h) or 1
        out_i = torch.empty((n, bh, bw, images.shape[3]), dtype=images.dtype,
                            device=images.device)
        out_m = None
        if mask is not None:
            m = mask if mask.dim() == 3 else mask.unsqueeze(0)
            out_m = torch.empty((n, bh, bw), dtype=m.dtype, device=m.device)
        for i in range(n):
            b = boxes[i]
            x = min(sc(b["x"], sx, w), w - bw)
            y = min(sc(b["y"], sy, h), h - bh)
            out_i[i] = images[i, y:y + bh, x:x + bw, :]
            if out_m is not None:
                out_m[i] = m[i, y:y + bh, x:x + bw]
        if out_m is None:
            out_m = torch.zeros((n, bh, bw))

        info = f"{w}x{h} -> {bw}x{bh}"
        if abs(sx - 1.0) > 0.01:
            info += f" (stream is {sx:.2f}x the cropped clip; boxes scaled to match)"
        logging.info("H3ApplyCrop: %s", info)
        return {"ui": {"h3char": [info]}, "result": (out_i, out_m, info)}


class H3PreviewMaskCrop:
    """See the mask and the crop box on the footage, before spending a render.

    Three overlays, and the third is the one worth having:

      mask          your per-pixel mask, tinted over the frames. Tells you whether
                    segmentation actually found the subject.
      latent mask   THE MASK AS THE MODEL RECEIVES IT. Reduced to 16px cells and
                    unioned over the VAE's (1,4,4,4,4) frame grouping, then blown
                    back up. This is what decides what gets regenerated, and it is
                    much coarser than the mask you drew — a thin occluder crossing
                    the subject can vanish into a cell, and a fast-moving edge
                    smears across the frames sharing a latent step. Watching this
                    one is how you find out why an edge did not land where you
                    expected.
      crop box      the rectangle H3 Subject Crop chose, drawn per frame. Shows
                    whether the box is too tight, whether it moves, and whether
                    the subject ever escapes it.

    `dilate` mirrors H3 Mask Inpaint, so set it to the same value and the latent
    overlay is exactly what that node will build.

    Feed the output to a video preview rather than an image preview — a single
    frame tells you almost nothing about tracking or smear.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "opacity": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0,
                        "step": 0.05}),
        }, "optional": {
            "mask": ("MASK",),
            "crop_data": ("H3_CROP",),
            "show_mask": ("BOOLEAN", {"default": True,
                          "tooltip": "Your pixel mask, in green."}),
            "show_latent_mask": ("BOOLEAN", {"default": True,
                                 "tooltip": "What the model actually gets, in red — "
                                            "16px cells, unioned over each latent "
                                            "frame's group. Coarser than yours, and "
                                            "the difference is what bites."}),
            "show_crop_box": ("BOOLEAN", {"default": True}),
            "dilate": ("INT", {"default": 0, "min": 0, "max": 16,
                       "tooltip": "Match H3 Mask Inpaint's dilate to see exactly what "
                                  "it will build."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "go"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Overlay the mask, the mask AS THE MODEL RECEIVES IT, and the crop "
                   "box on the source frames. Send it to a video preview.")

    def go(self, images, opacity, mask=None, crop_data=None, show_mask=True,
           show_latent_mask=True, show_crop_box=True, dilate=0):
        from .timing import frame_groups, video_latent_t

        out = images.clone()[..., :3]
        n, ih, iw = out.shape[0], out.shape[1], out.shape[2]
        a = float(opacity)
        notes = []

        def tint(sel, rgb):
            """sel: [N,H,W] in 0..1. Paint rgb over the frames where it is set."""
            s = (sel.to(out.device, out.dtype) * a).unsqueeze(-1)
            col = torch.tensor(rgb, device=out.device, dtype=out.dtype)
            out.mul_(1.0 - s).add_(s * col)

        if mask is not None:
            m = (mask if mask.dim() == 3 else mask.unsqueeze(0)).float()
            if m.shape[-2:] != (ih, iw):
                raise ValueError(f"mask is {m.shape[-1]}x{m.shape[-2]} but the clip is "
                                 f"{iw}x{ih}.")
            if show_latent_mask:
                lt = video_latent_t(n)
                sizes = frame_groups(lt)
                if sum(sizes) != n:
                    notes.append(f"{n} frames is not a legal run, so the latent view "
                                 f"is approximate — H3 would use {sum(sizes)}")
                    sizes = None
                if sizes:
                    q = m.unsqueeze(0).unsqueeze(0)                 # [1,1,N,H,W]
                    q = Fn.max_pool3d(q, (1, 16, 16), stride=(1, 16, 16))
                    q = torch.stack([g.amax(dim=2) for g in
                                     torch.split(q, sizes, dim=2)], dim=2)
                    if dilate > 0:
                        k = dilate * 2 + 1
                        q = Fn.max_pool3d(q, (1, k, k), stride=1,
                                          padding=(0, dilate, dilate))
                    # back to pixels: 16x spatially, then each latent frame repeated
                    # across the pixel frames it covers
                    q = q.repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2)
                    q = q[0, 0]                                     # [lt,h,w]
                    q = q.repeat_interleave(
                        torch.tensor(sizes, device=q.device), dim=0)
                    q = q[:n, :ih, :iw]
                    tint(q, (1.0, 0.15, 0.15))
                    cover = float(q.mean())
                    notes.append(f"latent mask covers {cover * 100:.1f}% of the frame "
                                 f"({lt} latent frames)")
            if show_mask:
                tint(m, (0.15, 1.0, 0.35))
                notes.append(f"pixel mask covers {float(m.mean()) * 100:.1f}%")

        if crop_data is not None and show_crop_box:
            boxes = crop_data["boxes"]
            sx = iw / float(crop_data["image_width"])
            sy = ih / float(crop_data["image_height"])
            t = max(1, int(round(min(ih, iw) / 260)))       # visible at any size
            for i in range(min(n, len(boxes))):
                b = boxes[i]
                x0 = max(0, min(int(b["x"] * sx), iw - 1))
                y0 = max(0, min(int(b["y"] * sy), ih - 1))
                x1 = max(x0 + 1, min(int((b["x"] + b["width"]) * sx), iw))
                y1 = max(y0 + 1, min(int((b["y"] + b["height"]) * sy), ih))
                out[i, y0:y0 + t, x0:x1, :] = 1.0
                out[i, max(y0, y1 - t):y1, x0:x1, :] = 1.0
                out[i, y0:y1, x0:x0 + t, :] = 1.0
                out[i, y0:y1, max(x0, x1 - t):x1, :] = 1.0
            b = boxes[0]
            notes.append(f"crop {b['width']}x{b['height']}, "
                         f"{b['width'] * b['height'] / float(iw * ih) * 100:.0f}% of "
                         f"the frame")

        info = "; ".join(notes) or "nothing to overlay — wire a mask or crop_data"
        logging.info("H3PreviewMaskCrop: %s", info)
        return {"ui": {"h3char": [info]}, "result": (out.clamp(0, 1), info)}


NODE_CLASS_MAPPINGS = {"H3SubjectCrop": H3SubjectCrop,
                       "H3SubjectUncrop": H3SubjectUncrop,
                       "H3ApplyCrop": H3ApplyCrop,
                       "H3PreviewMaskCrop": H3PreviewMaskCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SubjectCrop": "H3 Subject Crop",
                              "H3SubjectUncrop": "H3 Subject Uncrop",
                              "H3ApplyCrop": "H3 Apply Crop (depth / pose / etc)",
                              "H3PreviewMaskCrop": "H3 Preview Mask + Crop"}
