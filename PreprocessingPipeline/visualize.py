"""
Visualization tool for foreground masks.

Run on a sample of dataset crops to verify the mask generator is producing
sensible decompositions before integrating into a training pipeline.

Usage (programmatic):
    from foreground_attention import ForegroundMaskGenerator
    from foreground_attention.visualize import visualize_masks
    gen = ForegroundMaskGenerator()
    visualize_masks(
        image_paths=['/path/a.jpg', '/path/b.jpg'],
        classes=['trafficsignal', 'Crosswalk'],
        generator=gen,
        output_dir='./viz_foreground/',
    )

Usage (CLI):
    python -m foreground_attention.visualize \\
        --images path1.jpg path2.jpg --classes trafficsignal Crosswalk \\
        --output-dir ./viz_foreground/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .mask_generator import ForegroundMaskGenerator


def _stack_horiz(images: List[np.ndarray], pad: int = 4) -> np.ndarray:
    """Concatenate images horizontally with a separator. All resized to same H."""
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    target_h = max(im.shape[0] for im in images)
    sep = np.full((target_h, pad, 3), 255, dtype=np.uint8)
    rows = []
    for i, im in enumerate(images):
        if im.shape[0] != target_h:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(im)
            new_w = max(1, int(im.shape[1] * target_h / im.shape[0]))
            pil = pil.resize((new_w, target_h))
            im = np.asarray(pil)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        rows.append(im)
        if i < len(images) - 1:
            rows.append(sep)
    return np.concatenate(rows, axis=1)


def _mask_to_rgb_vis(mask: np.ndarray, base_image: np.ndarray) -> np.ndarray:
    """Visualise a (H, W) float mask by dimming the image outside the mask
    and tinting the masked region yellow.

    Result: image is dim/dark where mask=0, bright with yellow tint where
    mask=1. Easy to see at a glance which region the part-token attends to.
    """
    H, W = mask.shape
    base = base_image
    if base.shape[:2] != (H, W):
        from PIL import Image as PILImage
        base_pil = PILImage.fromarray(base).resize((W, H))
        base = np.asarray(base_pil)
    base = base.astype(np.float32)

    # alpha: 0.25 outside mask, up to 1.0 inside
    alpha = 0.25 + 0.75 * mask
    out = base * alpha[..., None]

    # Add a yellow tint where the mask is strong
    tint_strength = mask * 60.0
    out[..., 0] = np.clip(out[..., 0] + tint_strength, 0, 255)
    out[..., 1] = np.clip(out[..., 1] + tint_strength, 0, 255)
    return out.astype(np.uint8)


def visualize_masks(
    image_paths: List[str],
    classes: List[str],
    generator: ForegroundMaskGenerator,
    output_dir: str,
    save_individual: bool = True,
) -> None:
    """
    For each (image_path, class) pair, save a visualization showing:
        original | foreground binary | mask 0 overlay | mask 1 overlay | mask 2 overlay
    """
    if len(image_paths) != len(classes):
        raise ValueError(
            f"image_paths and classes must be same length: "
            f"{len(image_paths)} vs {len(classes)}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = []
    for path, cls in zip(image_paths, classes):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"WARNING: cannot open {path}: {e}")
            continue
        rgb = np.asarray(img)

        masks = generator(img, cls=cls)
        # foreground binary (only meaningful if not fallback)
        fb = generator.get_foreground_binary(img)
        fb_vis = (fb * 255).astype(np.uint8)
        if fb_vis.ndim == 2:
            fb_vis = np.stack([fb_vis] * 3, axis=-1)

        panels = [rgb, fb_vis]
        for r in range(masks.shape[0]):
            panels.append(_mask_to_rgb_vis(masks[r], rgb))
        row = _stack_horiz(panels, pad=4)

        if save_individual:
            label = f"{Path(path).stem}_{cls}.png"
            Image.fromarray(row).save(out_dir / label)

        grid_rows.append(row)

    if grid_rows:
        max_w = max(r.shape[1] for r in grid_rows)
        padded = []
        for r in grid_rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        sep = np.full((4, max_w, 3), 200, dtype=np.uint8)
        stacked = []
        for i, r in enumerate(padded):
            stacked.append(r)
            if i < len(padded) - 1:
                stacked.append(sep)
        grid = np.concatenate(stacked, axis=0)
        Image.fromarray(grid).save(out_dir / "_all.png")
        print(f"Saved combined grid to {out_dir / '_all.png'}")
    print(f"Wrote {len(grid_rows)} visualizations to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--output-dir", default="./viz_foreground/")
    parser.add_argument("--clahe-clip", type=float, default=3.0)
    parser.add_argument("--blur-sigma", type=float, default=5.0)
    parser.add_argument("--min-side-decomp", type=int, default=24)
    args = parser.parse_args()

    gen = ForegroundMaskGenerator(
        clahe_clip_limit=args.clahe_clip,
        boundary_blur_sigma=args.blur_sigma,
        min_side_for_decomp=args.min_side_decomp,
    )
    visualize_masks(args.images, args.classes, gen, args.output_dir)


if __name__ == "__main__":
    main()