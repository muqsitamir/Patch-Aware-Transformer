"""
Foreground mask generator.

Pipeline:
  1. CLAHE on the L channel of the LAB-converted image (enhances contrast
     for OTSU without distorting colour for the model).
  2. OTSU thresholding on the enhanced grayscale -> binary foreground.
  3. Connected-component analysis -> keep largest non-background component.
  4. Bounding box of largest component -> divide into N horizontal strips.
  5. Each strip becomes a soft (Gaussian-blurred) attention mask.

Fallback to whole-image masks when:
  - the class is in `fallback_classes` (e.g., crosswalks — panoramic, no parts)
  - min(H, W) < min_side_for_decomp (resolution too low for meaningful split)
  - OTSU produces a degenerate mask (all-zero or all-one)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# OpenCV is the only non-stdlib runtime dep besides numpy/PIL.
# We import lazily so the package can be imported on a machine without cv2
# for inspecting docs/types, but the generator will fail fast at construction
# if cv2 isn't available.
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


class ForegroundMaskGenerator:
    """
    Generate `num_regions` foreground-aware region masks per input image.

    Args:
        num_regions:           how many region masks to produce. Default 3.
        clahe_clip_limit:      CLAHE clip limit on the L channel.
        clahe_tile_grid:       CLAHE tile grid size (cv2 default is (8,8)).
        boundary_blur_sigma:   Gaussian sigma applied to each binary mask
                               before stacking, for soft boundaries. Set to
                               0 to keep masks strictly binary.
        min_side_for_decomp:   if min(H, W) is below this, fall back to
                               whole-image masks (no useful sub-structure).
        fallback_classes:      tuple of class names (lowercased before match)
                               that always get whole-image masks.

    Notes on the bbox split:
        With num_regions=3 and the default split_axis='horizontal', the
        foreground bounding box is divided into top/middle/bottom thirds.
        Each strip is intersected with the foreground mask, so the resulting
        per-region mask is "this strip AND foreground."
    """

    def __init__(
        self,
        num_regions: int = 3,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        boundary_blur_sigma: float = 5.0,
        min_side_for_decomp: int = 24,
        fallback_classes: Tuple[str, ...] = ("crosswalk",),
        split_axis: str = "horizontal",
    ):
        if cv2 is None:
            raise ImportError(
                "ForegroundMaskGenerator requires opencv-python. "
                "Install with `pip install opencv-python`."
            )
        if num_regions < 1:
            raise ValueError(f"num_regions must be >= 1, got {num_regions}")
        if split_axis not in ("horizontal", "vertical"):
            raise ValueError(
                f"split_axis must be 'horizontal' or 'vertical', got {split_axis}"
            )

        # Guard against the common bug of passing a string instead of a tuple:
        # `fallback_classes='crosswalk'` iterates as ['c','r','o','s',...].
        # Force the user to pass an iterable of strings, not a string.
        if isinstance(fallback_classes, str):
            raise TypeError(
                f"fallback_classes must be a tuple/list of strings, not a "
                f"single string. Did you mean fallback_classes=({fallback_classes!r},) "
                f"with a trailing comma?"
            )

        self.num_regions = num_regions
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.boundary_blur_sigma = float(boundary_blur_sigma)
        self.min_side_for_decomp = int(min_side_for_decomp)
        self.fallback_classes = tuple(c.lower() for c in fallback_classes)
        self.split_axis = split_axis

        # cache the CLAHE object (cv2 docs recommend reusing it)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )

    # -------- main entrypoint --------------------------------------------
    def __call__(self, pil_image, cls: str = "") -> np.ndarray:
        """
        Args:
            pil_image: PIL.Image.Image (RGB) or numpy array (HxWx3 uint8).
            cls:       class label string. Used for fallback decision.

        Returns:
            (num_regions, H, W) float32 array, values in [0, 1].
        """
        rgb = self._to_rgb_numpy(pil_image)
        H, W = rgb.shape[:2]

        # fallback path 1: explicit class fallback
        if cls and cls.lower() in self.fallback_classes:
            return self._whole_image_masks(H, W)

        # fallback path 2: resolution fallback
        if min(H, W) < self.min_side_for_decomp:
            return self._whole_image_masks(H, W)

        # main path
        try:
            fg_binary = self._compute_foreground_binary(rgb)
        except Exception:
            return self._whole_image_masks(H, W)

        # fallback path 3: degenerate OTSU output (uniform image)
        fg_pixel_count = int(fg_binary.sum())
        total_pixels = H * W
        if fg_pixel_count == 0 or fg_pixel_count == total_pixels:
            return self._whole_image_masks(H, W)
        # if the foreground is tiny, the bbox-split is too noisy
        if fg_pixel_count < (self.num_regions * 16):
            return self._whole_image_masks(H, W)

        bbox = self._foreground_bbox(fg_binary)
        if bbox is None:
            return self._whole_image_masks(H, W)

        masks = self._split_bbox_into_strips(fg_binary, bbox, H, W)
        masks = self._soften_boundaries(masks)
        return masks

    # -------- public helper for visualization ---------------------------
    def get_foreground_binary(self, pil_image) -> np.ndarray:
        """Return the binary foreground mask used internally (for debugging)."""
        rgb = self._to_rgb_numpy(pil_image)
        return self._compute_foreground_binary(rgb)

    # -------- internals --------------------------------------------------
    def _to_rgb_numpy(self, pil_image) -> np.ndarray:
        if Image is not None and isinstance(pil_image, Image.Image):
            arr = np.asarray(pil_image.convert("RGB"))
        else:
            arr = np.asarray(pil_image)
            if arr.ndim == 2:  # grayscale -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:  # RGBA -> RGB
                arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def _whole_image_masks(self, H: int, W: int) -> np.ndarray:
        """Return num_regions identical full-image (all-ones) masks."""
        return np.ones((self.num_regions, H, W), dtype=np.float32)

    def _compute_foreground_binary(self, rgb: np.ndarray) -> np.ndarray:
        """CLAHE on L, OTSU, morphological closing, return uint8 binary mask.

        The morphological closing step matters: many objects (signs with
        symbols inside, bins with handles) appear as MULTIPLE connected
        components after OTSU. Closing fills the small gaps so the entire
        object becomes one big region rather than many small ones.
        """
        # Convert to LAB and apply CLAHE to the L channel only.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[..., 0]
        L_eq = self._clahe.apply(L)

        # OTSU works on grayscale. We use the equalised L channel directly.
        _, fg = cv2.threshold(
            L_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # OTSU may pick the foreground or the background as "1" — we want
        # "1 = the smaller-area class" because objects are usually a subset
        # of the crop. If the >127 region is more than half the image, invert.
        if fg.mean() > 127:
            fg = 255 - fg

        # Morphological closing fills small gaps between disjoint pieces of
        # the same object. Kernel size scales with image size — we want it
        # big enough to bridge symbol-to-border gaps in signs but small
        # enough not to merge truly separate objects.
        H, W = fg.shape
        k = max(3, min(H, W) // 16)
        if k % 2 == 0:  # ensure odd
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        return (fg > 0).astype(np.uint8)

    def _foreground_bbox(self, binary: np.ndarray):
        """Return (y0, y1, x0, x1) covering ALL foreground pixels.

        We use the bbox of the entire foreground rather than the largest
        connected component, because multi-piece objects (sign+symbol+text)
        are common and all the pieces matter together.
        """
        if binary.sum() == 0:
            return None
        ys, xs = np.where(binary > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return (y0, y1, x0, x1)

    def _split_bbox_into_strips(
        self,
        fg_binary: np.ndarray,
        bbox,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Divide the bbox into num_regions strips along split_axis.

        Each strip is intersected with the foreground binary mask.
        Returns (num_regions, H, W) float32.
        """
        y0, y1, x0, x1 = bbox
        masks = np.zeros((self.num_regions, H, W), dtype=np.float32)

        if self.split_axis == "horizontal":
            # split top-to-bottom
            edges = np.linspace(y0, y1, self.num_regions + 1).astype(int)
            for i in range(self.num_regions):
                yA, yB = edges[i], edges[i + 1]
                strip = np.zeros((H, W), dtype=np.float32)
                strip[yA:yB, x0:x1] = 1.0
                masks[i] = strip * fg_binary.astype(np.float32)
        else:  # vertical
            edges = np.linspace(x0, x1, self.num_regions + 1).astype(int)
            for i in range(self.num_regions):
                xA, xB = edges[i], edges[i + 1]
                strip = np.zeros((H, W), dtype=np.float32)
                strip[y0:y1, xA:xB] = 1.0
                masks[i] = strip * fg_binary.astype(np.float32)

        return masks

    def _soften_boundaries(self, masks: np.ndarray) -> np.ndarray:
        """Optionally blur each mask channel for soft boundaries."""
        if self.boundary_blur_sigma <= 0:
            return masks
        blurred = np.empty_like(masks)
        # ksize must be odd and roughly 6*sigma for the kernel to capture the bell
        ksize = max(3, int(6 * self.boundary_blur_sigma) | 1)  # force odd
        for i in range(masks.shape[0]):
            blurred[i] = cv2.GaussianBlur(
                masks[i],
                (ksize, ksize),
                self.boundary_blur_sigma,
            )
        # clip to [0, 1] (Gaussian blur can produce small overshoots in fp32
        # but normally stays in range)
        return np.clip(blurred, 0.0, 1.0)