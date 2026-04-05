"""
V-JEPA sub-modules: encoder wrapper, predictor, and patch embedding.

The encoder wraps a timm ViT backbone to support partial-sequence forward
(only visible/unmasked patches), which is essential for masked prediction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """Convert video clips to patch token sequences with positional embeddings.

    Input:  (B, T, C, H, W)
    Output: (B, T * n_h * n_w, D)  with added positional embeddings
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 n_frames: int = 16, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.n_h = img_size // patch_size
        self.n_w = img_size // patch_size
        self.n_spatial = self.n_h * self.n_w
        self.n_tokens = n_frames * self.n_spatial

        # Patch projection: per-frame 2D convolution
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Separate spatial and temporal positional embeddings (additive)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_spatial, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, n_frames, embed_dim))

        self._init_pos_embed()

    def _init_pos_embed(self):
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T * n_h * n_w, D)
        """
        B, T = x.shape[:2]
        # Flatten batch and time for convolution
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, C, H, W)
        x = self.proj(x)                     # (B*T, D, n_h, n_w)
        D = x.shape[1]
        x = x.reshape(B, T, D, -1)           # (B, T, D, n_spatial)
        x = x.permute(0, 1, 3, 2)            # (B, T, n_spatial, D)

        # Add positional embeddings
        x = x + self.spatial_pos.unsqueeze(1)          # spatial: broadcast over T
        x = x + self.temporal_pos[:, :T].unsqueeze(2)  # temporal: broadcast over spatial

        x = x.reshape(B, T * self.n_spatial, D)  # (B, N_total, D)
        return x


class VJEPAEncoder(nn.Module):
    """Wraps a timm ViT to support partial-sequence forward.

    Standard ViTs process all patches. V-JEPA's student encoder only sees
    visible (unmasked) patches. This wrapper accesses the ViT's internals
    to achieve this.

    For non-ViT backbones (e.g., ConvNext), falls back to full forward + gather.
    """

    def __init__(self, backbone: nn.Module, patch_embed_3d: PatchEmbedding3D):
        super().__init__()
        self.patch_embed_3d = patch_embed_3d

        # Try to detect ViT internals for efficient partial forward
        self.is_vit = hasattr(backbone, "blocks") and hasattr(backbone, "norm")
        if self.is_vit:
            # Use ViT blocks directly, bypassing backbone's patch_embed
            self.blocks = backbone.blocks
            self.norm = backbone.norm
            self.embed_dim = backbone.embed_dim
        else:
            # Fallback: full forward, then gather visible positions
            self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        visible_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video clip
            visible_indices: (N_visible,) indices into flattened token sequence.
                If None, processes all tokens (teacher mode).

        Returns:
            (B, N_out, D) where N_out = N_visible or N_total
        """
        # Patch embedding: (B, N_total, D)
        tokens = self.patch_embed_3d(x)

        if self.is_vit:
            return self._forward_vit(tokens, visible_indices)
        return self._forward_fallback(tokens, visible_indices)

    def _forward_vit(
        self, tokens: torch.Tensor, visible_indices: torch.LongTensor | None
    ) -> torch.Tensor:
        """Efficient partial forward through ViT blocks."""
        if visible_indices is not None:
            # Gather only visible tokens
            B = tokens.shape[0]
            vis = visible_indices.unsqueeze(0).expand(B, -1)  # (B, N_vis)
            tokens = torch.gather(
                tokens, 1, vis.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            )  # (B, N_vis, D)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens

    def _forward_fallback(
        self, tokens: torch.Tensor, visible_indices: torch.LongTensor | None
    ) -> torch.Tensor:
        """Full forward, then optionally gather visible positions."""
        # Process all tokens through backbone blocks
        # (backbone expects (B, N, D) after our custom patch embed)
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks:
                tokens = block(tokens)
            if hasattr(self.backbone, "norm"):
                tokens = self.backbone.norm(tokens)
        # Gather if needed
        if visible_indices is not None:
            B = tokens.shape[0]
            vis = visible_indices.unsqueeze(0).expand(B, -1)
            tokens = torch.gather(
                tokens, 1, vis.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            )
        return tokens


class VJEPAPredictor(nn.Module):
    """Small transformer that predicts masked token representations.

    Takes visible encoder outputs + learnable mask tokens at masked positions,
    with positional information, and outputs predictions for all positions.

    Args:
        embed_dim: dimension of encoder output / mask tokens
        predictor_dim: internal dimension of predictor transformer
        depth: number of transformer layers
        n_heads: number of attention heads
        n_tokens: total number of spatiotemporal tokens
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        predictor_dim: int = 512,
        depth: int = 6,
        n_heads: int = 8,
        n_tokens: int = 3136,  # 16 * 14 * 14
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.n_tokens = n_tokens

        # Project encoder output to predictor dimension
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for all token positions
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=n_heads,
                dim_feedforward=predictor_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dimension
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        visible_indices: torch.LongTensor,
        masked_indices: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: (B, N_visible, D) encoder output for visible patches
            visible_indices: (N_visible,) positions in the full sequence
            masked_indices: (N_masked,) positions of masked patches

        Returns:
            (B, N_total, D) predicted representations for all positions
        """
        B = visible_tokens.shape[0]
        N_vis = visible_indices.shape[0]
        N_mask = masked_indices.shape[0]
        N_total = N_vis + N_mask

        # Project visible tokens to predictor dim
        vis_proj = self.input_proj(visible_tokens)  # (B, N_vis, predictor_dim)
        dtype = vis_proj.dtype

        # Create mask tokens
        mask_tokens = self.mask_token.to(dtype).expand(B, N_mask, -1)  # (B, N_mask, predictor_dim)

        # Assemble full sequence: place visible and mask tokens at correct positions
        full_seq = torch.zeros(
            B, N_total, self.predictor_dim,
            device=visible_tokens.device, dtype=dtype,
        )

        vis_idx = visible_indices.unsqueeze(0).expand(B, -1)   # (B, N_vis)
        mask_idx = masked_indices.unsqueeze(0).expand(B, -1)    # (B, N_mask)

        full_seq.scatter_(1, vis_idx.unsqueeze(-1).expand(-1, -1, self.predictor_dim), vis_proj)
        full_seq.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, self.predictor_dim), mask_tokens)

        # Add positional embeddings
        full_seq = full_seq + self.pos_embed[:, :N_total].to(dtype)

        # Transformer forward
        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)

        # Project back to encoder dimension
        return self.output_proj(full_seq)  # (B, N_total, D)
