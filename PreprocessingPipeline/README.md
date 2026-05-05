# foreground_attention

Drop-in package for foreground-aware part attention in Vision Transformer ReID models.

## What this is

A preprocessing pipeline + an architecture-agnostic attention bias function for re-ID tasks where input crops contain significant background.

The standard PAT approach divides each input crop into 3 fixed horizontal stripes and treats them as part-tokens. For non-person objects (traffic signs, bins, etc.), these stripes don't fit — top-third is often sky, bottom-third is often pole. This package replaces the fixed stripes with **content-defined attention regions** derived from CLAHE+OTSU foreground masking, then exposes those regions as additive attention biases that any transformer can use.

## What's in the box

```
foreground_attention/
    __init__.py            -- public API
    mask_generator.py      -- ForegroundMaskGenerator
    paired_transforms.py   -- (image, mask) augmentation helpers
    cache.py               -- MaskCache (disk cache for generated masks)
    attention_bias.py      -- apply_foreground_bias() and helpers
    visualize.py           -- visualization tool
    tests/                 -- standalone tests
    README.md              -- this file
    INTEGRATION.md         -- model-side integration guide (PAT, ViT-B)
```

## Installation

This is a folder, not a pip package. Drop it into your project and import.

Dependencies:
- `numpy`
- `opencv-python` (for CLAHE + OTSU; install via `pip install opencv-python`)
- `Pillow`
- `torch` and `torchvision` (only needed for paired transforms and the bias function; mask generator alone works without)

## Quick start

### 1. Generate masks

```python
from foreground_attention import ForegroundMaskGenerator
from PIL import Image

gen = ForegroundMaskGenerator(
    num_regions=3,
    clahe_clip_limit=3.0,
    boundary_blur_sigma=5.0,
    min_side_for_decomp=24,
    fallback_classes=('crosswalk',),
)

img = Image.open('some_sign.jpg')
masks = gen(img, cls='trafficsignal')
# masks: (3, H, W) float32 array, values in [0, 1]
```

For `cls` in `fallback_classes` or images with `min(H, W) < min_side_for_decomp`,
all 3 masks are returned as all-ones (whole-image attention — model falls back
to standard global self-attention).

### 2. Visualize before training

```python
from foreground_attention.visualize import visualize_masks

visualize_masks(
    image_paths=['a.jpg', 'b.jpg', ...],
    classes=['trafficsignal', 'Crosswalk', ...],
    generator=gen,
    output_dir='./viz/',
)
```

Inspect the output. Each row shows: original | foreground binary | mask 0 | mask 1 | mask 2.

**If 20%+ of the visualizations look wrong, do not proceed to training.** OTSU
fails on uniform crops (e.g., backs-of-signs, flat bin walls). The fallback
mechanism handles many such cases by detecting tiny foreground area, but it's
not a substitute for visual inspection.

### 3. Cache masks across epochs

```python
from foreground_attention import MaskCache

cache = MaskCache(
    cache_dir='/path/to/cache/',
    generator_config={
        'num_regions': 3,
        'clahe_clip_limit': 3.0,
        'boundary_blur_sigma': 5.0,
        'min_side_for_decomp': 24,
    },
)

# In your dataloader:
def get_masks(image_path, cls):
    return cache.get_or_compute(
        image_path,
        lambda: gen(Image.open(image_path), cls=cls),
    )
```

The cache is keyed by image relpath + a hash of the generator config. Changing
the config invalidates the cache.

### 4. Paired augmentation

Geometric augmentations (flip, resize, crop) MUST be applied identically to both
image and masks, or alignment breaks.

```python
from foreground_attention import (
    PairedImageMaskTransform, PairedHFlip, PairedResize, PairedPadCrop,
)
import torchvision.transforms as T

paired = PairedImageMaskTransform(
    paired_ops=[
        PairedHFlip(p=0.5),
        PairedResize((256, 128)),
        PairedPadCrop(padding=10, crop_size=(256, 128)),
    ],
    image_only_ops=[
        T.ColorJitter(0.15, 0.15, 0.10, 0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ],
)

image_tensor, masks_tensor = paired(pil_image, masks_numpy)
# image_tensor: (3, H, W) torch FloatTensor
# masks_tensor: (3, H, W) torch FloatTensor
```

### 5. Apply the attention bias inside your transformer

This is the architecture-coupled step. See `INTEGRATION.md` for full examples
in PAT and ViT-B. The high-level call — note that because the directory name
contains a hyphen you must add it to `sys.path` before importing:

```python
import sys, os
_PIPELINE_DIR = os.path.abspath('/path/to/Preprocessing-Pipeline')
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from attention_bias import apply_foreground_bias, downsample_masks_to_patch_grid

# Inside your forward pass, you have:
#   masks: (B, 3, H, W) tensor
# Convert to patch-grid resolution:
patch_masks = downsample_masks_to_patch_grid(masks, patch_size=16)
# patch_masks: (B, 3, num_patches)

# Inside each transformer block's attention (BEFORE softmax):
attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
attn_scores = apply_foreground_bias(
    attn_scores,
    patch_masks,
    part_token_indices=[1, 2, 3],  # positions of part tokens in seq
    patch_token_indices=list(range(4, N)),  # patch token positions
    bias_value=-1e4,
)
attn = attn_scores.softmax(dim=-1)
```

For efficiency, precompute the full bias matrix once per forward pass and add
it to attn_scores in each layer (this is what the PAT integration does):

```python
from attention_bias import make_attn_bias_matrix

bias_matrix = make_attn_bias_matrix(
    patch_masks,
    part_token_indices=[1, 2, 3],
    patch_token_indices=list(range(4, N)),
    num_tokens=N,
    bias_value=-1e4,
)
# (B, 1, N, N) — broadcasts across heads; cast to attn dtype under AMP

# In every transformer block (pre-softmax):
attn = (q @ k.transpose(-2, -1)) * scale + bias_matrix.to(dtype=attn.dtype)
```

## Configuration recommendations

Based on Urban Elements ReID 2026 EDA:

```python
ForegroundMaskGenerator(
    num_regions=3,                       # matches PAT's 3 part-tokens
    clahe_clip_limit=3.0,                # cv2 default
    boundary_blur_sigma=5.0,             # soft mask boundaries to avoid
                                         # discrete artifacts that the model
                                         # could exploit as fake features
    min_side_for_decomp=24,              # ~1/3 of dataset is below this;
                                         # they all fall back to whole-image
    fallback_classes=('crosswalk',),     # crosswalks are panoramic, fixed
                                         # horizontal split is meaningless
)
```

Bias value:
- `-1e4` (default): hard mask. Patches outside the region get effectively zero
  attention weight from that part-token.
- `-2.0`: soft mask. Attention is biased away from outside-region patches but
  the model can override if outside-region patches are very informative.

Use the soft setting if your foreground masks are noisy. Use the hard setting if
they're reliable.

## Caveats

1. **OTSU is not perfect.** On uniform crops (backs-of-signs, flat surfaces) it
   produces tiny disconnected foreground blobs. The package detects this and
   falls back to whole-image masks, but the detection isn't perfect. **Visualize
   first.**

2. **Mask masking is not augmentation.** The masks define which patches each
   part-token attends to. They don't replace or modify pixels in the image. The
   model still sees the full image; only the attention pattern changes.

3. **Caching matters.** Mask generation costs ~10-50 ms per image. Without
   caching, you'd add 30+ hours of pure CPU work to a 60-epoch run.

4. **Disable PC_LOSS in PAT.** The original PAT has a part-classification
   auxiliary loss that assumes the parts are class-discriminative on their own.
   This doesn't hold when part assignments are content-defined. Disable it.

5. **Pre-softmax, not post-softmax.** The bias must be applied to the raw
   attention logits before softmax. Adding a bias to softmax outputs is wrong.

## License / sharing

This is internal research code. If you share with collaborators, point them
to `INTEGRATION.md` for the model-side hookup details.
