"""
Architecture-agnostic attention bias function.

This module is the contract between the foreground_attention package and
any specific transformer architecture. As long as the caller can:

  1. Identify which token positions in the sequence are "part-query" tokens
     (positions where you want region-specific attention).
  2. Identify which token positions are image patch tokens.
  3. Provide pre-softmax attention scores.

…this function adds an additive bias that pushes each part-query token to
attend to its assigned region and away from patches outside that region.

This contract is satisfied by:

  - PAT (Part-Aware Transformer): part-tokens 1, 2, 3 in the sequence
    [CLS, p1, p2, p3, patch_0, ..., patch_N].
  - Vanilla ViT-B with manually-added part-query tokens: insert N learnable
    tokens after CLS, treat them analogously.
  - Any other architecture where you can introduce additional query tokens.

The function is intentionally pure (no in-place mutation) for easier debugging.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def downsample_masks_to_patch_grid(
    masks: Tensor,
    patch_size: int,
    output_layout: str = "flat",
) -> Tensor:
    """
    Downsample pixel-resolution masks to patch-grid resolution.

    Args:
        masks:         (B, R, H, W) float tensor in [0, 1].
        patch_size:    transformer patch size (e.g., 16 for ViT-B/16).
        output_layout: "flat" -> (B, R, H_p * W_p)
                       "grid" -> (B, R, H_p, W_p)

    Returns:
        Patch-grid mask, average-pooled. Values in [0, 1].
    """
    if masks.dim() != 4:
        raise ValueError(f"expected (B, R, H, W), got {tuple(masks.shape)}")
    B, R, H, W = masks.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"image size ({H}, {W}) must be divisible by patch_size {patch_size}"
        )

    H_p, W_p = H // patch_size, W // patch_size
    pooled = torch.nn.functional.avg_pool2d(
        masks,
        kernel_size=patch_size,
        stride=patch_size,
    )
    # pooled shape: (B, R, H_p, W_p)
    if output_layout == "flat":
        return pooled.reshape(B, R, H_p * W_p)
    elif output_layout == "grid":
        return pooled
    else:
        raise ValueError(
            f"output_layout must be 'flat' or 'grid', got {output_layout!r}"
        )


def apply_foreground_bias(
    attn_scores: Tensor,
    patch_masks: Tensor,
    part_token_indices: List[int],
    patch_token_indices: List[int],
    bias_value: float = -1e4,
) -> Tensor:
    """
    Add an additive bias to attention scores so that each part-query token
    preferentially attends to patches inside its assigned mask.

    Args:
        attn_scores:         pre-softmax attention logits of shape
                             (B, num_heads, num_tokens, num_tokens).
        patch_masks:         per-part attention mask of shape
                             (B, num_part_tokens, num_patches),
                             values in [0, 1]. 1.0 = attend, 0.0 = ignore.
                             num_part_tokens MUST equal len(part_token_indices).
                             num_patches    MUST equal len(patch_token_indices).
        part_token_indices:  positions in the token sequence of the
                             part-query tokens. Order matches the leading
                             axis of patch_masks.
        patch_token_indices: positions of the image patch tokens.
        bias_value:          large negative value applied where mask is 0.
                             Default -1e4 effectively excludes those patches
                             from softmax. Use a smaller magnitude (e.g.,
                             -2.0) if you want a soft bias the model can
                             override when needed.

    Returns:
        Modified attention scores of the same shape as input. NOT in-place;
        the input is not mutated.

    Notes:
        - Bias is broadcast across attention heads (same bias for every head).
        - The CLS token (and any other non-part, non-patch tokens) are
          unaffected — their attention rows/columns are not modified.
        - Bias only affects the part-token rows of attn_scores, columns at
          the patch positions. Other rows and columns are untouched.

    Shape reference:
        - attn_scores[b, h, i, j] is "how much does token i attend to token j
          (in batch b, head h)" before softmax.
        - We modify entries where i ∈ part_token_indices and j ∈
          patch_token_indices.
    """
    # input validation
    if attn_scores.dim() != 4:
        raise ValueError(
            f"attn_scores must be (B, H, N, N); got {tuple(attn_scores.shape)}"
        )
    if patch_masks.dim() != 3:
        raise ValueError(
            f"patch_masks must be (B, R, P); got {tuple(patch_masks.shape)}"
        )
    B, num_heads, N1, N2 = attn_scores.shape
    if N1 != N2:
        raise ValueError(
            f"attn_scores last two dims must be equal (square), got {N1} vs {N2}"
        )
    Bm, R, P = patch_masks.shape
    if B != Bm:
        raise ValueError(f"batch size mismatch: attn={B}, masks={Bm}")
    if R != len(part_token_indices):
        raise ValueError(
            f"patch_masks has {R} part dims but {len(part_token_indices)} "
            f"part_token_indices were given"
        )
    if P != len(patch_token_indices):
        raise ValueError(
            f"patch_masks has {P} patch dims but {len(patch_token_indices)} "
            f"patch_token_indices were given"
        )

    # bias is large-negative where mask=0, zero where mask=1.
    # Shape: (B, R, P)
    bias = (1.0 - patch_masks) * bias_value

    # We'll write into a clone; non-mutating contract.
    out = attn_scores.clone()

    # Convert indices to long tensors for advanced indexing
    part_idx = torch.tensor(part_token_indices, device=attn_scores.device, dtype=torch.long)
    patch_idx = torch.tensor(patch_token_indices, device=attn_scores.device, dtype=torch.long)

    # Build the broadcast: we want out[:, :, part_idx[r], patch_idx[p]]
    # to receive bias[:, r, p], summed over a single head dim (broadcast).
    #
    # Approach: extract the (B, H, R, P) sub-tensor corresponding to part rows
    # and patch columns, add the bias broadcast over heads, write back.

    # Index into rows (part) first, then columns (patch).
    # out[:, :, part_idx, :] -> shape (B, H, R, N)
    # then we want the patch columns: [:, :, :, patch_idx] -> (B, H, R, P)
    # advanced indexing with two long tensors gets tricky; use explicit gather.
    #
    # Cleanest approach: build a full (B, H, N, N) bias tensor that's zero
    # everywhere except at (part_idx, patch_idx) cells. Memory cost is
    # B*H*N*N which is the same as attn_scores itself — acceptable.

    full_bias = torch.zeros_like(attn_scores)
    # (B, H, R, P) <- broadcast bias (B, R, P) over heads
    full_bias[:, :, part_idx[:, None], patch_idx[None, :]] = bias.unsqueeze(1)
    out = out + full_bias
    return out


def make_attn_bias_matrix(
    patch_masks: Tensor,
    part_token_indices: List[int],
    patch_token_indices: List[int],
    num_tokens: int,
    bias_value: float = -1e4,
) -> Tensor:
    """
    Convenience: precompute a full (B, 1, N, N) attention bias matrix that
    can be added to attn_scores once per attention layer (broadcasting across
    heads).

    This is more efficient than calling apply_foreground_bias inside every
    layer — you compute the bias once per forward pass and add it to scores
    in each attention block.

    Returns:
        (B, 1, N, N) tensor — shape compatible with broadcasting against
        (B, num_heads, N, N) attention scores.
    """
    if patch_masks.dim() != 3:
        raise ValueError(f"patch_masks must be (B, R, P), got {tuple(patch_masks.shape)}")
    B, R, P = patch_masks.shape
    if R != len(part_token_indices):
        raise ValueError(
            f"patch_masks has {R} part dims but {len(part_token_indices)} indices given"
        )
    if P != len(patch_token_indices):
        raise ValueError(
            f"patch_masks has {P} patch dims but {len(patch_token_indices)} indices given"
        )

    bias = (1.0 - patch_masks) * bias_value  # (B, R, P)

    full_bias = torch.zeros(B, 1, num_tokens, num_tokens,
                            device=patch_masks.device, dtype=patch_masks.dtype)
    part_idx = torch.tensor(part_token_indices, device=patch_masks.device, dtype=torch.long)
    patch_idx = torch.tensor(patch_token_indices, device=patch_masks.device, dtype=torch.long)
    # (B, 1, R, P) <- (B, R, P) unsqueezed
    full_bias[:, 0:1, part_idx[:, None], patch_idx[None, :]] = bias.unsqueeze(1)
    return full_bias
