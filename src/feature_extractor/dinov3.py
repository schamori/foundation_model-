"""DINOv3 feature extractors (HuggingFace transformers).

Requires transformers >= 5.3.0 for DINOv3 support.
Both ViT and ConvNeXt variants available.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from . import FeatureExtractor, register_extractor


class _DINOv3Base(FeatureExtractor):
    """Shared logic for DINOv3 variants."""

    hf_model_id: str = ""
    input_size: int = 224
    num_register_tokens: int = 4  # DINOv3 ViT uses 4 register tokens

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load_model(self, device: torch.device) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        self.model = AutoModel.from_pretrained(
            self.hf_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()

    def preprocess(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
        out = self.model(pixel_values=pixel_values)
        # Use pooler_output if available (CLS-based), else mean-pool patch tokens
        if out.pooler_output is not None:
            feats = out.pooler_output
        else:
            # Skip CLS + register tokens, mean-pool patches
            skip = 1 + self.num_register_tokens
            feats = out.last_hidden_state[:, skip:, :].mean(dim=1)
        return feats.float().cpu().numpy().astype(np.float16)


# -- ViT variants --

@register_extractor
class DINOv3ViTS(_DINOv3Base):
    name = "dinov3-vits"
    hf_model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    embed_dim = 384


@register_extractor
class DINOv3ViTB(_DINOv3Base):
    name = "dinov3-vitb"
    hf_model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    embed_dim = 768


@register_extractor
class DINOv3ViTL(_DINOv3Base):
    name = "dinov3-vitl"
    hf_model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    embed_dim = 1024


# -- ConvNeXt variants --

class _DINOv3ConvNeXt(_DINOv3Base):
    """ConvNeXt DINOv3 — no register tokens, uses pooler_output."""
    num_register_tokens = 0


@register_extractor
class DINOv3ConvNeXtTiny(_DINOv3ConvNeXt):
    name = "dinov3-convnext-tiny"
    hf_model_id = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    embed_dim = 768


@register_extractor
class DINOv3ConvNeXtSmall(_DINOv3ConvNeXt):
    name = "dinov3-convnext-small"
    hf_model_id = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    embed_dim = 768


@register_extractor
class DINOv3ConvNeXtBase(_DINOv3ConvNeXt):
    name = "dinov3-convnext-base"
    hf_model_id = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    embed_dim = 1024


@register_extractor
class DINOv3ConvNeXtLarge(_DINOv3ConvNeXt):
    name = "dinov3-convnext-large"
    hf_model_id = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
    embed_dim = 1536
