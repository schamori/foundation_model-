"""
Named experiment configurations.

Each experiment is a subclass of Config that overrides a few fields.
Comment/uncomment entries in get_experiment_configs() to select which to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class Baseline(Config):
    """Vanilla DINO — no temporal neighbors, no cross-video pairs."""

    EXPERIMENT_NAME: str = "baseline"
    use_cross_video_pairs: bool = False
    temporal_neighbor_range: int = 0


@dataclass
class WithTemporal(Config):
    """DINO + temporal neighbor crops (±2 frames)."""

    EXPERIMENT_NAME: str = "with_temporal"
    use_cross_video_pairs: bool = False
    temporal_neighbor_range: int = 2


@dataclass
class WithCrossVideo(Config):
    """DINO + cross-video pair mining (no temporal)."""

    EXPERIMENT_NAME: str = "with_crossvideo"
    temporal_neighbor_range: int = 0
    use_cross_video_pairs: bool = True
    cross_video_factor: float = 3.0


@dataclass
class Full(Config):
    """DINO + temporal + cross-video — all features enabled."""

    EXPERIMENT_NAME: str = "full"
    temporal_neighbor_range: int = 2
    use_cross_video_pairs: bool = True
    cross_video_factor: float = 3.0


@dataclass
class VJEPABaseline(Config):
    """V-JEPA baseline — no motion masking, no cross-video."""

    EXPERIMENT_NAME: str = "vjepa_baseline"
    ssl_method: str = "vjepa"
    use_motion_guided_masking: bool = False
    lambda_cross: float = 0.0
    lambda_affinity: float = 0.0


@dataclass
class VJEPAWithMotion(Config):
    """V-JEPA + motion-guided masking."""

    EXPERIMENT_NAME: str = "vjepa_motion"
    ssl_method: str = "vjepa"
    use_motion_guided_masking: bool = True
    lambda_cross: float = 0.0


@dataclass
class VJEPAWithCrossVideo(Config):
    """V-JEPA + cross-video Hungarian matching (no motion)."""

    EXPERIMENT_NAME: str = "vjepa_crossvideo"
    ssl_method: str = "vjepa"
    use_motion_guided_masking: bool = False
    lambda_cross: float = 0.3
    use_cross_video_pairs: bool = True


@dataclass
class VJEPAFull(Config):
    """V-JEPA — all features: motion masking + affinity + cross-video + SFDR."""

    EXPERIMENT_NAME: str = "vjepa_full"
    ssl_method: str = "vjepa"
    use_motion_guided_masking: bool = True
    use_cross_video_pairs: bool = True


def get_experiment_configs() -> list[Config]:
    """Return the list of experiments to run. Comment/uncomment to select."""
    return [
        Baseline(),
        WithTemporal(),
        WithCrossVideo(),
        Full(),
        VJEPABaseline(),
        VJEPAWithMotion(),
        VJEPAWithCrossVideo(),
        VJEPAFull(),
    ]
