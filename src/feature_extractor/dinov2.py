"""DINOv2 feature extractors (HuggingFace transformers)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from . import FeatureExtractor, register_extractor


class _DINOv2Base(FeatureExtractor):
    """Shared logic for all DINOv2 variants."""

    hf_model_id: str = ""
    input_size: int = 224

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load_model(self, device: torch.device) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        self.model = AutoModel.from_pretrained(self.hf_model_id).to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()

    def preprocess(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            out = self.model(pixel_values=pixel_values)
        feats = out.last_hidden_state.mean(dim=1)
        return feats.float().cpu().numpy().astype(np.float16)


@register_extractor
class DINOv2Small(_DINOv2Base):
    name = "dinov2-small"
    hf_model_id = "facebook/dinov2-small"
    embed_dim = 384


@register_extractor
class DINOv2Base(_DINOv2Base):
    name = "dinov2-base"
    hf_model_id = "facebook/dinov2-base"
    embed_dim = 768


@register_extractor
class DINOv2Large(_DINOv2Base):
    name = "dinov2-large"
    hf_model_id = "facebook/dinov2-large"
    embed_dim = 1024


@register_extractor
class DINOv2Giant(_DINOv2Base):
    name = "dinov2-giant"
    hf_model_id = "facebook/dinov2-giant"
    embed_dim = 1536
