"""
Augmentation pipelines for DINO and V-JEPA training.

- MultiCropAugmentation: standard single-image DINO multi-crop
- PoolAwareMultiCropAugmentation: LemonFM-style pool of images (cross-video + temporal)
- VideoClipAugmentation: consistent spatial transform across a video clip (V-JEPA)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torchvision.transforms as T
from PIL import Image

if TYPE_CHECKING:
    from ..config import Config

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_global_transform_1(size: int = 224, scale: tuple[float, float] = (0.32, 1.0)) -> T.Compose:
    """Global crop view 1 — always applies Gaussian blur."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_global_transform_2(size: int = 224, scale: tuple[float, float] = (0.32, 1.0)) -> T.Compose:
    """Global crop view 2 — light blur + solarization."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
        T.RandomSolarize(threshold=256, p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_local_transform(size: int = 96, scale: tuple[float, float] = (0.05, 0.32)) -> T.Compose:
    """Local crop — smaller region, lighter augmentation."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiCropAugmentation:
    """Config-driven multi-crop augmentation for DINO-style training."""

    def __init__(self, cfg: Config):
        self.n_global = cfg.n_global_crops
        self.n_local = cfg.n_local_crops
        self.global_t1 = build_global_transform_1(cfg.image_size, cfg.global_crop_scale)
        self.global_t2 = build_global_transform_2(cfg.image_size, cfg.global_crop_scale)
        # Local crop size = image_size * 96/224 (preserves original ratio)
        local_size = max(32, int(cfg.image_size * 96 / 224))
        self.local_t = build_local_transform(local_size, cfg.local_crop_scale)

    def __call__(self, img: Image.Image) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns (global_crops, local_crops) as tensor lists."""
        globals_ = []
        for i in range(self.n_global):
            t = self.global_t1 if i == 0 else self.global_t2
            globals_.append(t(img))
        locals_ = [self.local_t(img) for _ in range(self.n_local)]
        return globals_, locals_


class PoolAwareMultiCropAugmentation:
    """LemonFM-style augmentation: crops drawn from a pool of images.

    The pool contains the anchor frame, temporal neighbors, and cross-video
    matches. Teacher global crops come from one pool member, student crops
    from different members — forcing cross-patient invariance through the
    standard DINO loss.

    Falls back to standard single-image behavior when pool has one image.
    """

    def __init__(self, cfg: Config):
        self.n_global = cfg.n_global_crops
        self.n_local = cfg.n_local_crops
        self.global_t1 = build_global_transform_1(cfg.image_size, cfg.global_crop_scale)
        self.global_t2 = build_global_transform_2(cfg.image_size, cfg.global_crop_scale)
        local_size = max(32, int(cfg.image_size * 96 / 224))
        self.local_t = build_local_transform(local_size, cfg.local_crop_scale)

    def __call__(self, pool: list[Image.Image]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns (global_crops, local_crops) from pool members."""
        if len(pool) == 1:
            img = pool[0]
            globals_ = []
            for i in range(self.n_global):
                t = self.global_t1 if i == 0 else self.global_t2
                globals_.append(t(img))
            locals_ = [self.local_t(img) for _ in range(self.n_local)]
            return globals_, locals_

        # Multi-image pool: diversify crop sources
        globals_ = []
        for i in range(self.n_global):
            src = random.choice(pool)
            t = self.global_t1 if i == 0 else self.global_t2
            globals_.append(t(src))

        locals_ = [self.local_t(random.choice(pool)) for _ in range(self.n_local)]
        return globals_, locals_


class VideoClipAugmentation:
    """Consistent spatial augmentation for V-JEPA video clips.

    Applies the same spatial transform (crop, flip, color jitter) to all
    frames in a clip so temporal coherence is preserved.

    Returns a tensor of shape (T, C, H, W).
    """

    def __init__(self, cfg: Config):
        self.size = cfg.image_size
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, clip: list[Image.Image]) -> torch.Tensor:
        """Apply consistent spatial transform to all clip frames."""
        # Sample transform parameters once for the entire clip
        i, j, h, w = T.RandomResizedCrop.get_params(
            clip[0], scale=(0.3, 1.0), ratio=(0.75, 1.35)
        )
        flip = random.random() < 0.5

        frames = []
        for frame in clip:
            frame = T.functional.resized_crop(
                frame, i, j, h, w, [self.size, self.size],
                interpolation=T.InterpolationMode.BICUBIC,
            )
            if flip:
                frame = T.functional.hflip(frame)
            t = T.functional.to_tensor(frame)
            t = self.normalize(t)
            frames.append(t)

        return torch.stack(frames)  # (T, C, H, W)


def build_augmentation(cfg: Config):
    """Factory: return the right augmentation for the configured SSL method."""
    if cfg.ssl_method == "vjepa":
        return VideoClipAugmentation(cfg)
    if cfg.use_cross_video_pairs or cfg.temporal_neighbor_range > 0:
        return PoolAwareMultiCropAugmentation(cfg)
    return MultiCropAugmentation(cfg)
