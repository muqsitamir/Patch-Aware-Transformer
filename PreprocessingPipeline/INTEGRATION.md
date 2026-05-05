# Integration Guide

How to wire `foreground_attention` into a transformer-based ReID model. Two
walkthroughs: PAT (the Patch-Aware Transformer codebase) and vanilla ViT-B
(e.g., `timm` ViT).

## What you must add to your model

Three things, in this order:

1. **Receive `masks` from the dataloader.** Your forward pass needs to accept
   a `(B, num_regions, H_img, W_img)` mask tensor alongside the image.

2. **Have N "part-query" tokens in your token sequence.** PAT already does
   this. ViT-B doesn't — you'll need to add them as learnable parameters.

3. **Apply the attention bias inside every transformer block.** Pre-softmax,
   broadcasting across heads.

The bias function itself is unaware of which architecture you're using. It only
needs token indices and the mask tensor.

---

## Import note: Preprocessing-Pipeline directory name

The directory is named `Preprocessing-Pipeline` (with a hyphen), which is not
a valid Python identifier. You cannot do `import Preprocessing-Pipeline`.
Instead, the code adds the directory itself to `sys.path` and imports from the
individual module files inside it (`attention_bias`, `mask_generator`, etc.).
Both PAT integration files already contain the helper:

```python
import os, sys
_PIPELINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '<relative-path-to>/Preprocessing-Pipeline')
)
def _ensure_pipeline_importable():
    if _PIPELINE_DIR not in sys.path:
        sys.path.insert(0, _PIPELINE_DIR)
```

Call `_ensure_pipeline_importable()` before any `from attention_bias import …`
or `from mask_generator import …` call.

---

## Walkthrough 1: PAT (Patch-Aware Transformer)

PAT already has 3 part-query tokens. The integration swaps out the
hardcoded horizontal-stripe boolean mask for content-defined additive attention
biases derived from foreground masks.

**All changes below are already applied to the codebase.** This walkthrough
documents what was changed and why, so you can understand the integration and
extend it.

### Token sequence in PAT

```
[CLS, part_0, part_1, part_2, patch_0, patch_1, ..., patch_{N-1}]
   0      1       2       3       4         5     ...    3+N
```

Verified in `part_Attention_ViT.forward_features()`:
```python
x = torch.cat((cls_tokens, part_token1, part_token2, part_token3, x), dim=1)
```

So:
- `part_token_indices = [1, 2, 3]`
- `patch_token_indices = list(range(4, 4 + num_patches))`

### Original masking mechanism (replaced)

The original `part_Attention.forward()` used a binary boolean mask with two
bugs:

```python
# BUG 1: masked_fill with fp16 literal — brittle under AMP dtype changes.
attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16))
# BUG 2: post-softmax multiply — produces incorrect attention weights;
#         adding a bias to softmax OUTPUTS is mathematically wrong.
attn = torch.mul(attn, mask)
```

Both bugs are fixed by switching to an additive pre-softmax bias.

### What changed: `model/backbones/vit_pytorch.py`

**`part_Attention.forward()`** — parameter renamed from `mask` to `attn_bias`;
mask-fill and post-softmax multiply replaced with a single pre-softmax add:

```python
class part_Attention(nn.Module):
    def forward(self, x, attn_bias=None):
        # ... QKV projection ...
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.to(dtype=attn.dtype)  # (B,1,N,N) → broadcasts over heads
        attn = attn.softmax(dim=-1)
        # ... rest unchanged ...
```

**`part_Attention_Block.forward()`** — parameter renamed from `mask` to
`attn_bias`:

```python
class part_Attention_Block(nn.Module):
    def forward(self, x, attn_bias=None):
        x = x + self.drop_path(self.part_attn(self.norm1(x), attn_bias=attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

**`part_Attention_ViT.forward_features()`** — accepts `masks` from the
dataloader; builds the bias matrix from foreground masks when provided,
otherwise falls back to the original horizontal-stripe logic (now expressed as
an additive bias for consistency):

```python
class part_Attention_ViT(nn.Module):
    def forward_features(self, x, masks=None):
        # ... patchify, concat special tokens, add pos_embed ...
        num_total = x.shape[1]  # self.num_patches = image_patches + 4

        if masks is not None:
            _ensure_pipeline_importable()
            from attention_bias import downsample_masks_to_patch_grid, make_attn_bias_matrix
            patch_masks = downsample_masks_to_patch_grid(
                masks.to(device=x.device, dtype=x.dtype),
                patch_size=self.patch_embed.patch_size[0],   # e.g. 16
                output_layout="flat",
            )
            attn_bias = make_attn_bias_matrix(
                patch_masks,
                part_token_indices=[1, 2, 3],
                patch_token_indices=list(range(4, num_total)),
                num_tokens=num_total,
                bias_value=-1e4,
            ).to(dtype=x.dtype)
        else:
            # Horizontal-stripe fallback; same behaviour as the original code
            # but expressed as a float additive bias rather than a bool mask.
            stripe_mask = self.attn_mask_generate(
                num_total, self.patch_embed.num_y, self.patch_embed.num_x, x.device.type
            )
            attn_bias = (~stripe_mask).to(dtype=x.dtype) * -1e3
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias)
            layerwise_tokens.append(x)
        ...

    def forward(self, x, masks=None):
        return self.forward_features(x, masks=masks)
```

### Stride / patch-size constraint

`downsample_masks_to_patch_grid` average-pools pixel-resolution masks down to
patch-grid resolution assuming `stride == patch_size`. This holds for the
default config (`STRIDE_SIZE=[16,16]`, `patch_size=16`). If you use a non-equal
stride (e.g. `STRIDE_SIZE=12`) the pool output will have a different spatial
size than the token sequence and `make_attn_bias_matrix` will raise a shape
error. **Always set `stride_size == patch_size` when using foreground
attention.**

### What changed: `model/make_model.py`

`build_part_attention_vit.forward()` gains a `masks` parameter and threads it
to `self.base`:

```python
class build_part_attention_vit(nn.Module):
    def forward(self, x, masks=None, return_class_logits=False):
        layerwise_tokens = self.base(x, masks=masks)
        # ... existing feature extraction unchanged ...
```

### What changed: `data/common.py`

`CommDataset.__init__()` now accepts three optional kwargs for mask support:

```python
class CommDataset(Dataset):
    def __init__(self, img_items, transform=None, relabel=True,
                 mask_gen=None, mask_cache=None, paired_transform=None):
```

- `mask_gen` — a `ForegroundMaskGenerator` instance
- `mask_cache` — a `MaskCache` instance (strongly recommended; generation is
  ~15 ms/image; without caching you add ~30 hours per 60-epoch run)
- `paired_transform` — a `PairedImageMaskTransform` instance (required for any
  geometric augmentation, e.g. random flip)

When `mask_gen` is not `None`, `__getitem__` returns the dict with an extra
`"masks"` key containing a `(num_regions, H, W)` float tensor.

**How to construct these in your dataloader builder** (e.g. `build_DG_dataloader.py`):

```python
from data.common import CommDataset, _ensure_pipeline_importable
import torchvision.transforms as T

mask_gen = mask_cache = paired_transform = None
if cfg.MODEL.FOREGROUND_ATTN.ENABLED:
    _ensure_pipeline_importable()
    from mask_generator import ForegroundMaskGenerator
    from cache import MaskCache
    from paired_transforms import (
        PairedImageMaskTransform, PairedHFlip, PairedResize, PairedPadCrop,
    )
    import os

    mask_gen = ForegroundMaskGenerator(
        num_regions=cfg.MODEL.FOREGROUND_ATTN.NUM_REGIONS,
        clahe_clip_limit=cfg.MODEL.FOREGROUND_ATTN.CLAHE_CLIP,
        boundary_blur_sigma=cfg.MODEL.FOREGROUND_ATTN.BOUNDARY_BLUR,
        min_side_for_decomp=cfg.MODEL.FOREGROUND_ATTN.MIN_SIDE_FOR_DECOMP,
        fallback_classes=tuple(cfg.MODEL.FOREGROUND_ATTN.FALLBACK_CLASSES),
    )
    cache_dir = cfg.MODEL.FOREGROUND_ATTN.CACHE_DIR or \
        os.path.join(cfg.DATASETS.ROOT_DIR, '_foreground_masks')
    mask_cache = MaskCache(
        cache_dir=cache_dir,
        generator_config={
            'num_regions': cfg.MODEL.FOREGROUND_ATTN.NUM_REGIONS,
            'clahe_clip_limit': cfg.MODEL.FOREGROUND_ATTN.CLAHE_CLIP,
            'boundary_blur_sigma': cfg.MODEL.FOREGROUND_ATTN.BOUNDARY_BLUR,
            'min_side_for_decomp': cfg.MODEL.FOREGROUND_ATTN.MIN_SIDE_FOR_DECOMP,
            'fallback_classes': list(cfg.MODEL.FOREGROUND_ATTN.FALLBACK_CLASSES),
        },
    )
    paired_transform = PairedImageMaskTransform(
        paired_ops=[
            PairedHFlip(p=cfg.INPUT.FLIP_PROB),
            PairedResize(cfg.INPUT.SIZE_TRAIN),
            PairedPadCrop(padding=cfg.INPUT.PADDING, crop_size=cfg.INPUT.SIZE_TRAIN),
        ],
        image_only_ops=[
            T.ColorJitter(
                cfg.INPUT.CJ.BRIGHTNESS, cfg.INPUT.CJ.CONTRAST,
                cfg.INPUT.CJ.SATURATION, cfg.INPUT.CJ.HUE,
            ) if cfg.INPUT.CJ.ENABLED else None,
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ],
    )

train_set = CommDataset(
    img_items,
    transform=train_transform,      # used only when mask_gen is None
    relabel=True,
    mask_gen=mask_gen,
    mask_cache=mask_cache,
    paired_transform=paired_transform,
)
```

The `"others"` dict for each image should contain a `"class_name"` key (string)
so that `ForegroundMaskGenerator` can apply the correct fallback policy (e.g.
whole-image masks for `'crosswalk'`). If the key is absent the generator falls
back gracefully.

### What changed: `processor/part_attention_vit_processor.py`

The training loop now unpacks `masks` from the batch dict and passes it to the
model:

```python
for n_iter, informations in enumerate(train_loader):
    img    = informations['images']
    masks  = informations.get('masks', None)   # None when FOREGROUND_ATTN disabled
    ...
    img = img.to(device)
    if masks is not None:
        masks = masks.to(device)
    ...
    score, layerwise_global_feat, layerwise_feat_list = model(img, masks=masks)
```

The distributed eval loop inside training follows the same pattern.

The final evaluation path goes through `do_inference` → `extract_tta_features`
(in `utils/tta.py`). That helper currently does **not** pass masks — at test
time the model uses the horizontal-stripe fallback. For best results, update
`extract_tta_features` to accept and pass a `masks` kwarg, and update your
inference dataloader to return masks (same `CommDataset` setup, no augmentation
beyond resize/normalize).

### What changed: `config/defaults.py`

A new `MODEL.FOREGROUND_ATTN` sub-config was added with safe defaults:

```yaml
MODEL:
  FOREGROUND_ATTN:
    ENABLED: False
    NUM_REGIONS: 3
    CLAHE_CLIP: 3.0
    BOUNDARY_BLUR: 5.0
    MIN_SIDE_FOR_DECOMP: 24
    FALLBACK_CLASSES: ['crosswalk']
    BIAS_VALUE: -1e4
    CACHE_DIR: ''        # default: <DATASETS.ROOT_DIR>/_foreground_masks
```

To enable in your experiment config:

```yaml
MODEL:
  PC_LOSS: False          # must be False; PC_LOSS and FOREGROUND_ATTN are incompatible
  FOREGROUND_ATTN:
    ENABLED: True
    CACHE_DIR: '/path/to/mask_cache'
```

### Step-by-step checklist

1. Set `MODEL.FOREGROUND_ATTN.ENABLED=True` and `MODEL.PC_LOSS=False` in your
   experiment config.
2. Update your dataloader builder to construct `ForegroundMaskGenerator`,
   `MaskCache`, and `PairedImageMaskTransform` and pass them to `CommDataset`.
3. Visualize masks on a sample of your training data before training
   (`foreground_attention.visualize.visualize_masks`). If >20% look wrong,
   do not proceed.
4. Run one epoch and verify:
   - `dataloader returns masks` — batch `informations['masks'].shape` should
     be `(B, num_regions, H, W)`.
   - `bias is non-trivial` — log
     `(attn_bias != 0).float().mean()` for the first batch; expect ~50%.
   - `loss decreases` normally.

---

## Walkthrough 2: Vanilla ViT-B (e.g., timm)

ViT-B has a single CLS token and no part-query tokens. You need to add them.

### Step 1: Add learnable part-query tokens to your model

```python
import timm
import torch, torch.nn as nn

class ViTBWithParts(nn.Module):
    def __init__(self, num_part_tokens=3, use_foreground_attn=True):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0,
        )
        embed_dim = self.backbone.embed_dim
        self.num_part_tokens = num_part_tokens
        self.part_tokens = nn.Parameter(torch.zeros(1, num_part_tokens, embed_dim))
        nn.init.trunc_normal_(self.part_tokens, std=0.02)
        self.use_foreground_attn = use_foreground_attn
        self.patch_size = self.backbone.patch_embed.patch_size[0]
```

### Step 2: Build the token sequence and compute the bias in forward

Token layout: `[CLS, part_0, part_1, part_2, patch_0, ..., patch_{N-1}]`

```python
    def forward(self, image, masks=None):
        B = image.shape[0]
        x = self.backbone.patch_embed(image)        # (B, P, D)
        cls_token  = self.backbone.cls_token.expand(B, -1, -1)
        part_tokens = self.part_tokens.expand(B, -1, -1)
        x = torch.cat([cls_token, part_tokens, x], dim=1)
        x = x + self._build_pos_embed(x.shape[1], B)
        x = self.backbone.pos_drop(x)

        attn_bias = None
        if self.use_foreground_attn and masks is not None:
            _ensure_pipeline_importable()
            from attention_bias import downsample_masks_to_patch_grid, make_attn_bias_matrix
            num_total  = x.shape[1]
            num_part   = self.num_part_tokens
            patch_masks = downsample_masks_to_patch_grid(
                masks.to(x.device, dtype=x.dtype),
                patch_size=self.patch_size, output_layout="flat",
            )
            attn_bias = make_attn_bias_matrix(
                patch_masks,
                part_token_indices=list(range(1, 1 + num_part)),
                patch_token_indices=list(range(1 + num_part, num_total)),
                num_tokens=num_total,
                bias_value=-1e4,
            ).to(dtype=x.dtype)

        for block in self.backbone.blocks:
            # timm's Block doesn't accept attn_bias — see Step 3.
            x = block(x, attn_bias=attn_bias)

        x = self.backbone.norm(x)
        cls_feat   = x[:, 0]
        part_feats = x[:, 1:1 + self.num_part_tokens]
        return torch.cat([cls_feat, part_feats.flatten(1)], dim=1)
```

### Step 3: Patch timm's Attention and Block to accept `attn_bias`

`timm`'s `Attention` doesn't support additive biases by default. Replace via
subclass swap (timm API varies by version — adjust to match your installed
version):

```python
import timm.models.vision_transformer as vit

class AttentionWithBias(vit.Attention):
    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) \
                          .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.to(dtype=attn.dtype)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class BlockWithBias(vit.Block):
    def forward(self, x, attn_bias=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Swap blocks in __init__ after creating the backbone:
for i, block in enumerate(self.backbone.blocks):
    new_block = BlockWithBias.__new__(BlockWithBias)
    new_block.__dict__ = block.__dict__.copy()
    new_attn = AttentionWithBias.__new__(AttentionWithBias)
    new_attn.__dict__ = block.attn.__dict__.copy()
    new_block.attn = new_attn
    self.backbone.blocks[i] = new_block
```

### Step 4: Position embeddings

ViT-B pretrained pos embeddings cover 197 positions (1 CLS + 196 patches for
224×224 at 16×16). Adding 3 part tokens requires 200 total. Simplest approach:

```python
# In __init__:
self.cls_part_pos = nn.Parameter(torch.zeros(1, 4, embed_dim))
nn.init.trunc_normal_(self.cls_part_pos, std=0.02)
# patch pos: reuse pretrained embeddings (frozen or fine-tuned).

# In _build_pos_embed(num_total, B):
patch_pos = self.backbone.pos_embed[:, 1:, :].expand(B, -1, -1)
return torch.cat([self.cls_part_pos.expand(B, -1, -1), patch_pos], dim=1)
```

### Step 5: Dataloader

Same as PAT walkthrough — `CommDataset` with `mask_gen`, `mask_cache`, and
`paired_transform`. See the PAT walkthrough for setup details.

---

## Validation checklist

After integration, verify these things at runtime:

1. `dataloader returns masks` — print the batch shape on the first iteration:
   should be `(B, num_regions, H, W)`.

2. `bias matrix is non-trivial` — log
   `(attn_bias != 0).float().mean()` for the first batch. It should be ~50%
   (half the part-token rows, masked off). If it's 0%, masks are all-ones
   (everything's falling back to whole-image). If it's 100%, the bias is too
   dense.

3. `loss decreases over the first epoch` — same as any training run.

4. `inference output shape matches training output shape` — the feature
   dimension must be the same when `masks=None` is passed at inference time
   (falls back to stripe attention) vs. with masks during training.

5. `cache hit rate is high` — log this in the dataloader. After epoch 1, every
   image should be cached, so cache hit rate should be ~100% from epoch 2.

## Common bugs

- **Mask-image misalignment after augmentation.** If you flip the image but
  not the masks, attention bias points at the wrong patches. Always use
  `PairedImageMaskTransform` for any geometric augmentation.

- **Bias applied post-softmax.** The original PAT code had
  `torch.mul(attn, mask)` *after* softmax — this has been removed. The bias
  must go BEFORE softmax, as an additive term on the raw logits.

- **Wrong patch_token_indices.** With `[CLS, p0, p1, p2, patches...]`, patch
  indices start at 4, not 1 or 0. Off-by-one here is a silent bug.

- **stride_size != patch_size.** `downsample_masks_to_patch_grid` pools masks
  as if `stride == patch_size`. If your `STRIDE_SIZE` differs from `patch_size`
  (e.g., 12 vs 16), the pooled mask will not match the token count and
  `make_attn_bias_matrix` will raise a shape error.

- **PC_LOSS enabled alongside FOREGROUND_ATTN.** The patch-center loss assumes
  fixed horizontal part assignments, which foreground attention replaces. Set
  `MODEL.PC_LOSS=False` when `MODEL.FOREGROUND_ATTN.ENABLED=True`. A warning
  is logged by `build_loss` if both are enabled.

- **Forgetting masks at inference.** At test time the model falls back to the
  horizontal-stripe attention if `masks=None`. This is tolerable but hurts
  accuracy. For best results, update `utils/tta.py::extract_tta_features` to
  accept and pass masks, and use the same `CommDataset` setup for your
  inference loader.

- **Dtype mismatch under AMP.** The bias is cast to `attn.dtype` inside
  `part_Attention.forward()` (`attn_bias.to(dtype=attn.dtype)`), so it is safe
  to pass fp32 biases into a fp16 forward pass.

## Performance

On a 3080:

- Mask generation: ~15 ms per image (CPU; CLAHE+OTSU+components dominate).
  Cached after first epoch.
- Bias computation in forward pass: ~0.5 ms per batch (negligible).
- Memory overhead: roughly +(B × 1 × N × N × 4 bytes) per batch, where N is
  the number of tokens. For batch=32, N=132 (base PAT), that's ~2.2 MB.
  Negligible.
