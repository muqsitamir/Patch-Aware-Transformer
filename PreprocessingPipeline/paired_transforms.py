"""
Paired image+mask augmentation pipeline.

Geometric ops (flip, resize, pad+crop) MUST be applied identically to image
and masks, otherwise mask-to-image alignment breaks. This module provides
those paired ops.

Color/normalization ops are applied to the image only — masks are spatial
indicators and shouldn't be normalized.

Conventions
-----------
- image arrives as PIL.Image.Image, leaves as torch.FloatTensor (C, H, W)
  after any final ToTensor/Normalize step the caller adds.
- masks arrive as np.ndarray (R, H, W) float32, leave as torch.FloatTensor
  (R, H, W).

Usage
-----
The simplest usage is via PairedImageMaskTransform, which wraps a list of
paired ops, then a list of image-only ops. See README.

But the individual paired ops are usable on their own — they take and return
(image, masks) tuples.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    TF = None


# Type alias for clarity
ImageType = Union["Image.Image", torch.Tensor]
MaskType = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masks_to_tensor(masks: MaskType) -> torch.Tensor:
    """Coerce masks (numpy array or tensor) to a (R, H, W) float tensor."""
    if isinstance(masks, np.ndarray):
        return torch.from_numpy(masks).float()
    return masks.float()


def _masks_to_numpy(masks: MaskType) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        return masks.detach().cpu().numpy()
    return masks


# ---------------------------------------------------------------------------
# Individual paired ops
# ---------------------------------------------------------------------------

class PairedHFlip:
    """Random horizontal flip applied to both image and masks."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: ImageType, masks: MaskType):
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, dims=[-1])
            else:
                image = TF.hflip(image)
            masks = _masks_to_tensor(masks)
            masks = torch.flip(masks, dims=[-1])
        return image, masks


class PairedResize:
    """
    Resize image and masks to (H, W). Image uses bilinear, masks use bilinear
    too (continuous values for soft boundaries).
    """

    def __init__(self, size: Tuple[int, int]):
        # Accept either (H, W) or a single int (square)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, image: ImageType, masks: MaskType):
        H, W = self.size
        # image
        if isinstance(image, torch.Tensor):
            image = F.interpolate(
                image.unsqueeze(0), size=(H, W), mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            image = TF.resize(image, [H, W])
        # masks: tensorize, then bilinear-interpolate
        masks = _masks_to_tensor(masks)
        if masks.dim() == 3:
            masks = F.interpolate(
                masks.unsqueeze(0), size=(H, W), mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise ValueError(f"masks must be (R, H, W), got shape {masks.shape}")
        return image, masks


class PairedPadCrop:
    """
    Pad both image and masks by `padding` pixels (constant for image, zero
    for masks), then take a random crop of size `crop_size`.

    Image padding uses fill=0 (or per-channel mean if you wire that in).
    """

    def __init__(self, padding: int, crop_size: Tuple[int, int]):
        self.padding = int(padding)
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))

    def __call__(self, image: ImageType, masks: MaskType):
        # We work in PIL/numpy domain for simplicity; the caller's outer
        # pipeline still does ToTensor afterwards.
        pad = self.padding
        Hc, Wc = self.crop_size

        # image
        if isinstance(image, torch.Tensor):
            image = TF.pad(image, [pad, pad, pad, pad], fill=0)
            _, Himg, Wimg = image.shape
        else:
            image = TF.pad(image, [pad, pad, pad, pad], fill=0)
            Wimg, Himg = image.size  # PIL: (W, H)

        # masks: pad with zeros (outside-foreground is background)
        masks_t = _masks_to_tensor(masks)
        masks_t = F.pad(masks_t, (pad, pad, pad, pad), value=0.0)

        # pick a random crop window
        max_y = Himg - Hc
        max_x = Wimg - Wc
        if max_y < 0 or max_x < 0:
            # padded image still smaller than crop; resize fallback
            return PairedResize((Hc, Wc))(image, masks_t)
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)

        if isinstance(image, torch.Tensor):
            image = image[:, y0:y0 + Hc, x0:x0 + Wc]
        else:
            image = image.crop((x0, y0, x0 + Wc, y0 + Hc))
        masks_t = masks_t[:, y0:y0 + Hc, x0:x0 + Wc]

        return image, masks_t


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

class PairedImageMaskTransform:
    """
    Convenience wrapper that runs a list of paired ops then a list of
    image-only ops.

    The image-only ops are typically `[ColorJitter(...), ToTensor(),
    Normalize(...)]`. They receive only the image and return only the image.
    The masks are returned at their post-paired-op shape.

    Args:
        paired_ops:     list of callables (image, masks) -> (image, masks)
        image_only_ops: list of callables image -> image (torchvision-style)
        ensure_tensor:  if True, convert masks to a torch tensor at the end.
    """

    def __init__(
        self,
        paired_ops: List = None,
        image_only_ops: List = None,
        ensure_tensor: bool = True,
    ):
        self.paired_ops = paired_ops or []
        self.image_only_ops = image_only_ops or []
        self.ensure_tensor = ensure_tensor

    def __call__(self, image, masks):
        for op in self.paired_ops:
            image, masks = op(image, masks)
        for op in self.image_only_ops:
            image = op(image)
        if self.ensure_tensor:
            masks = _masks_to_tensor(masks)
        return image, masks
