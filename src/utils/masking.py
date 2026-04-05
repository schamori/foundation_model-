"""
Spatiotemporal masking utilities for V-JEPA.

Generates tube masks (spatially contiguous across time) with optional
motion-guided biasing for surgical video pretraining.
"""

from __future__ import annotations

import torch


def generate_tube_mask(
    T: int,
    H_patches: int,
    W_patches: int,
    mask_ratio: float = 0.75,
    tube_length: int = 2,
) -> torch.BoolTensor:
    """Generate random spatiotemporal tube masks.

    Each tube spans `tube_length` consecutive frames at a fixed spatial
    position. Tubes are sampled uniformly at random.

    Args:
        T: number of temporal positions
        H_patches, W_patches: spatial patch grid dimensions
        mask_ratio: fraction of tokens to mask
        tube_length: temporal extent of each tube

    Returns:
        BoolTensor of shape (T, H_patches * W_patches) — True = masked
    """
    N_spatial = H_patches * W_patches
    n_tubes_temporal = T // tube_length
    total_tube_slots = n_tubes_temporal * N_spatial
    n_mask = int(total_tube_slots * mask_ratio)

    # Sample tube positions (spatial_idx, temporal_slot)
    perm = torch.randperm(total_tube_slots)[:n_mask]
    tube_spatial = perm % N_spatial
    tube_temporal = perm // N_spatial  # which temporal slot (0..n_tubes_temporal-1)

    mask = torch.zeros(T, N_spatial, dtype=torch.bool)
    for i in range(n_mask):
        s = tube_spatial[i]
        t_start = tube_temporal[i] * tube_length
        t_end = min(t_start + tube_length, T)
        mask[t_start:t_end, s] = True

    return mask


def motion_guided_tube_mask(
    T: int,
    H_patches: int,
    W_patches: int,
    mask_ratio: float = 0.75,
    tube_length: int = 2,
    motion_bias: torch.Tensor | None = None,
    strength: float = 2.0,
) -> torch.BoolTensor:
    """Generate tube masks biased toward high-motion regions.

    Args:
        motion_bias: (T, H_patches, W_patches) or (H_patches, W_patches) float
            tensor of motion magnitudes. Higher values → more likely to be masked.
            If None, falls back to uniform random masking.
        strength: exponent for motion bias (higher = stronger bias)

    Returns:
        BoolTensor of shape (T, H_patches * W_patches) — True = masked
    """
    if motion_bias is None:
        return generate_tube_mask(T, H_patches, W_patches, mask_ratio, tube_length)

    N_spatial = H_patches * W_patches
    n_tubes_temporal = T // tube_length
    total_tube_slots = n_tubes_temporal * N_spatial
    n_mask = int(total_tube_slots * mask_ratio)

    # Build per-tube-slot weights from motion
    if motion_bias.dim() == 2:
        # Static motion map — repeat across temporal slots
        spatial_weights = motion_bias.reshape(N_spatial).float()
        weights = spatial_weights.repeat(n_tubes_temporal)
    else:
        # Per-frame motion — average within each tube slot
        weights = []
        for t_slot in range(n_tubes_temporal):
            t_start = t_slot * tube_length
            t_end = min(t_start + tube_length, T)
            slot_motion = motion_bias[t_start:t_end].mean(dim=0)  # (H, W)
            weights.append(slot_motion.reshape(N_spatial))
        weights = torch.cat(weights)

    # Apply strength exponent and mix with uniform
    weights = weights.clamp(min=1e-8) ** strength
    uniform = torch.ones_like(weights)
    # 70% motion-guided, 30% uniform (ensures some background gets masked)
    weights = 0.7 * (weights / weights.sum()) + 0.3 * (uniform / uniform.sum())

    # Sample without replacement
    indices = torch.multinomial(weights, n_mask, replacement=False)

    tube_spatial = indices % N_spatial
    tube_temporal = indices // N_spatial

    mask = torch.zeros(T, N_spatial, dtype=torch.bool)
    for i in range(n_mask):
        s = tube_spatial[i]
        t_start = tube_temporal[i] * tube_length
        t_end = min(t_start + tube_length, T)
        mask[t_start:t_end, s] = True

    return mask


def mask_to_indices(
    mask: torch.BoolTensor,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Convert a boolean mask to visible and masked position indices.

    Args:
        mask: (T, N_spatial) bool tensor — True = masked

    Returns:
        (visible_indices, masked_indices) as 1D LongTensors into the
        flattened (T * N_spatial) sequence.
    """
    flat = mask.reshape(-1)
    masked_idx = torch.where(flat)[0]
    visible_idx = torch.where(~flat)[0]
    return visible_idx, masked_idx
