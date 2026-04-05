"""
Shared backbone utilities for all SSL methods.

Handles backbone creation (timm / HuggingFace), embedding dimension inference,
output pooling, and checkpoint loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import Config


def create_backbone(cfg: Config) -> nn.Module:
    """Create backbone from config using timm or HuggingFace transformers."""
    backbone_name = cfg.backbone

    # HuggingFace path (e.g. facebook/convnext-large-224)
    if "/" in backbone_name:
        from transformers import ConvNextModel
        return ConvNextModel.from_pretrained(backbone_name)

    # timm model name
    import timm
    return timm.create_model(backbone_name, pretrained=True, num_classes=0)


def get_embed_dim(backbone: nn.Module) -> int:
    """Infer embedding dimension from backbone."""
    if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_sizes"):
        return backbone.config.hidden_sizes[-1]
    if hasattr(backbone, "num_features"):
        return backbone.num_features
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = backbone(dummy)
        if hasattr(out, "pooler_output"):
            return out.pooler_output.shape[-1]
        if isinstance(out, torch.Tensor):
            return out.shape[-1]
    raise ValueError("Cannot infer embed_dim from backbone")


def pool_backbone_output(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward through backbone and pool to (B, D)."""
    out = backbone(x)
    if hasattr(out, "pooler_output"):
        return out.pooler_output.squeeze(-1).squeeze(-1)
    if isinstance(out, torch.Tensor):
        if out.dim() == 4:  # (B, C, H, W) — global avg pool
            return out.mean(dim=[2, 3])
        return out
    raise ValueError(f"Unexpected backbone output type: {type(out)}")


def load_backbone_from_checkpoint(cfg: Config, checkpoint_path: Path) -> nn.Module:
    """Load backbone weights from a training checkpoint."""
    backbone = create_backbone(cfg)

    if not checkpoint_path.exists():
        print(f"[backbone] Checkpoint not found: {checkpoint_path}, using pretrained weights")
        return backbone.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for key in ("teacher", "student"):
        if key in checkpoint:
            state = checkpoint[key]
            prefix = "backbone."
            backbone_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if backbone_state:
                backbone.load_state_dict(backbone_state, strict=False)
                print(f"[backbone] Loaded {key} backbone from {checkpoint_path.name} "
                      f"(epoch {checkpoint.get('epoch', '?')})")
                return backbone.eval()

    try:
        backbone.load_state_dict(checkpoint, strict=False)
        print(f"[backbone] Loaded backbone directly from {checkpoint_path.name}")
    except Exception as e:
        print(f"[backbone] Could not load checkpoint: {e}, using pretrained weights")

    return backbone.eval()
