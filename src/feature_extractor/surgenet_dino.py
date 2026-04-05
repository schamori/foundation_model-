"""SurgeNetDINO feature extractors (from rlpddejong/SurgeNetDINO).

Uses timm ViT backbones with surgical DINO-pretrained weights from
HuggingFace: rlpddejong/SurgeNetXL_DINOv1-v3

All models trained on SurgeNetXL surgical data.
"""

from __future__ import annotations

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from . import FeatureExtractor, register_extractor

_HF_BASE = "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main"

# (timm_model_id, img_size, patch_size, weight_filename, embed_dim)
_SURGENET_MODELS = {
    # DINOv1
    "surgenet-dinov1-vits": ("vit_small_patch16_224.dino", 224, 16,
                              "DINOv1_ViTs16_size224_SurgeNetXL.pth", 384),
    "surgenet-dinov1-vitb": ("vit_base_patch16_224.dino", 224, 16,
                              "DINOv1_ViTb16_size224_SurgeNetXL.pth", 768),
    # DINOv2
    "surgenet-dinov2-vits": ("vit_small_patch14_dinov2", 336, 14,
                              "DINOv2_ViTs14_size336_SurgeNetXL.pth", 384),
    "surgenet-dinov2-vitb": ("vit_base_patch14_dinov2", 336, 14,
                              "DINOv2_ViTb14_size336_SurgeNetXL.pth", 768),
    "surgenet-dinov2-vitl": ("vit_large_patch14_dinov2", 336, 14,
                              "DINOv2_ViTl14_size336_SurgeNetXL.pth", 1024),
    # DINOv3
    "surgenet-dinov3-vits": ("vit_small_patch16_dinov3.lvd1689m", 336, 16,
                              "DINOv3_ViTs16_size336_SurgeNetXL.pth", 384),
    "surgenet-dinov3-vitb": ("vit_base_patch16_dinov3.lvd1689m", 336, 16,
                              "DINOv3_ViTb16_size336_SurgeNetXL.pth", 768),
    "surgenet-dinov3-vitl": ("vit_large_patch16_dinov3.lvd1689m", 336, 16,
                              "DINOv3_ViTl16_size336_SurgeNetXL.pth", 1024),
}


class _SurgeNetDINOBase(FeatureExtractor):
    """Shared logic for SurgeNetDINO models loaded via timm."""

    timm_id: str = ""
    weight_file: str = ""
    _img_size: int = 224
    _patch_size: int = 16

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def load_model(self, device: torch.device) -> None:
        self.device = device
        url = f"{_HF_BASE}/{self.weight_file}?download=true"

        self.model = timm.create_model(
            self.timm_id,
            img_size=(self._img_size, self._img_size),
            patch_size=self._patch_size,
            num_classes=0,  # remove classification head, return features
        )

        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()

        # Build timm transform for this model
        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        data_cfg["input_size"] = (3, self._img_size, self._img_size)
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


def _make_surgenet_class(key: str, timm_id: str, img_size: int, patch_size: int,
                         weight_file: str, edim: int) -> type:
    """Dynamically create and register a SurgeNetDINO extractor class."""
    cls = type(
        f"SurgeNetDINO_{key.replace('-', '_')}",
        (_SurgeNetDINOBase,),
        {
            "name": key,
            "timm_id": timm_id,
            "weight_file": weight_file,
            "_img_size": img_size,
            "_patch_size": patch_size,
            "embed_dim": edim,
            "input_size": img_size,
        },
    )
    return register_extractor(cls)


# Register all SurgeNetDINO variants
for _key, (_timm_id, _img_size, _patch_size, _wf, _edim) in _SURGENET_MODELS.items():
    _make_surgenet_class(_key, _timm_id, _img_size, _patch_size, _wf, _edim)
