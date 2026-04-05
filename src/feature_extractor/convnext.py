"""ConvNext feature extractors (timm).

Uses pretrained ConvNext models from timm. The convnext-large variant
matches the backbone used for DINO pretraining in this project.
"""

from __future__ import annotations

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from . import FeatureExtractor, register_extractor


class _ConvNextBase(FeatureExtractor):
    """Shared logic for ConvNext models loaded via timm."""

    timm_id: str = ""

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def load_model(self, device: torch.device) -> None:
        self.device = device
        self.model = timm.create_model(self.timm_id, pretrained=True, num_classes=0)
        self.model = self.model.to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        self.transform = create_transform(**data_cfg, is_training=False)

    def preprocess(self, images: list) -> torch.Tensor:
        return torch.stack([self.transform(img.convert("RGB")) for img in images])

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
        feats = self.model(pixel_values)  # (B, D) with num_classes=0
        return feats.float().cpu().numpy().astype(np.float16)


@register_extractor
class ConvNextLarge(_ConvNextBase):
    name = "convnext-large"
    timm_id = "convnext_large"
    embed_dim = 1536
    input_size = 224
