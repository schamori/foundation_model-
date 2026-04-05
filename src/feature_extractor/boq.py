"""Bag-of-Queries (BoQ) feature extractors via torch.hub.

Uses pretrained BoQ models from https://github.com/amaralibey/Bag-of-Queries
Two variants: DINOv2 backbone (12288-dim) and ResNet50 backbone (16384-dim).
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T

from . import FeatureExtractor, register_extractor


class _BoQBase(FeatureExtractor):
    """Shared logic for BoQ models loaded via torch.hub."""

    backbone_name: str = ""
    output_dim: int = 0
    _img_size: tuple[int, int] = (322, 322)

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def load_model(self, device: torch.device) -> None:
        self.device = device
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries", "get_trained_boq",
            backbone_name=self.backbone_name, output_dim=self.output_dim,
        )
        self.model = self.model.to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self._img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, images: list) -> torch.Tensor:
        return torch.stack([self.transform(img.convert("RGB")) for img in images])

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
        descriptor, _attns = self.model(pixel_values)
        return descriptor.float().cpu().numpy().astype(np.float16)


@register_extractor
class BoQDINOv2(_BoQBase):
    name = "boq-dinov2"
    backbone_name = "dinov2"
    output_dim = 12288
    embed_dim = 12288
    input_size = 322
    _img_size = (322, 322)


@register_extractor
class BoQResNet50(_BoQBase):
    name = "boq-resnet50"
    backbone_name = "resnet50"
    output_dim = 16384
    embed_dim = 16384
    input_size = 384
    _img_size = (384, 384)
