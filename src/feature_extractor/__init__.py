"""
Feature extraction with swappable backbones.

Registry pattern: each model registers itself via @register_extractor.

Usage:
    from src.feature_extractor import get_extractor, list_extractors
    extractor = get_extractor("dinov2-base")
    extractor.extract(frames_dir, output_dir)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor(ABC):
    """Base interface for all feature extractors."""

    name: str = "base"
    embed_dim: int = 0
    input_size: int = 224

    @abstractmethod
    def load_model(self, device: torch.device) -> None:
        """Load model and processor onto device."""
        ...

    @abstractmethod
    def extract_features(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of preprocessed images.

        Args:
            pixel_values: (B, C, H, W) tensor on the model's device.

        Returns:
            (B, D) numpy array of float16 features.
        """
        ...

    @abstractmethod
    def preprocess(self, images: list) -> torch.Tensor:
        """Preprocess a list of PIL images into a batched tensor."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, type[FeatureExtractor]] = {}


def register_extractor(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    """Decorator to register a feature extractor."""
    _EXTRACTORS[cls.name] = cls
    return cls


def get_extractor(name: str) -> FeatureExtractor:
    """Instantiate a registered feature extractor by name."""
    if name not in _EXTRACTORS:
        available = ", ".join(sorted(_EXTRACTORS.keys()))
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
    return _EXTRACTORS[name]()


def list_extractors() -> list[str]:
    """List all registered extractor names."""
    return sorted(_EXTRACTORS.keys())


# Import to trigger registration
from . import boq, convnext, dinov2, dinov3, surgenet_dino  # noqa: E402, F401
