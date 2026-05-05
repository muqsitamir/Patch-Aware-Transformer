"""
foreground_attention — drop-in package for foreground-aware part attention
in Vision Transformer ReID models.

Designed for the Urban Elements ReID Challenge 2026, but architecture-agnostic.
Both PAT and ViT-B backbones can use the bias function and mask pipeline.

Public API
----------
ForegroundMaskGenerator  — build (3, H, W) region masks from a PIL image
PairedImageMaskTransform — augmentation pipeline that handles (image, mask) pairs
MaskCache                — disk cache for generated masks
apply_foreground_bias    — architecture-agnostic attention bias function
downsample_masks_to_patch_grid — pixel-resolution masks -> patch-grid masks

See README.md for usage and INTEGRATION.md for model-side hookup.
"""

from .mask_generator import ForegroundMaskGenerator
from .paired_transforms import (
    PairedImageMaskTransform,
    PairedHFlip,
    PairedResize,
    PairedPadCrop,
)
from .cache import MaskCache
from .attention_bias import (
    apply_foreground_bias,
    downsample_masks_to_patch_grid,
)

__version__ = "0.1.0"

__all__ = [
    "ForegroundMaskGenerator",
    "PairedImageMaskTransform",
    "PairedHFlip",
    "PairedResize",
    "PairedPadCrop",
    "MaskCache",
    "apply_foreground_bias",
    "downsample_masks_to_patch_grid",
]
