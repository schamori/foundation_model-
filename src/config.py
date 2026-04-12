"""
Base configuration dataclasses for foundation model pretraining.

Config names drive the entire output directory structure:
    output/<EXPERIMENT_NAME>/checkpoints/
    output/<EXPERIMENT_NAME>/eval/
    output/<EXPERIMENT_NAME>/config.json

All paths are auto-detected relative to PROJECT_ROOT (the repo root).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Auto-detect project root: src/config.py → go up one level
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Known data locations — tried in order, first existing path wins
_FRAMES_CANDIDATES = [
    Path("/media/HDD1/moritz/Extracted Frames/train"),

]

_EMBEDDINGS_CANDIDATES = [
    Path("/media/HDD1/moritz/Extracted Frames embeddings"),
    PROJECT_ROOT / "data" / "embeddings",
]


def _find_existing(candidates: list[Path], fallback: Path) -> Path:
    """Return the first candidate that exists, or fallback."""
    for p in candidates:
        if p.exists():
            return p
    return fallback


@dataclass
class DataConfig:
    """Dataset and data path configuration."""

    frames_root: Path = field(
        default_factory=lambda: _find_existing(_FRAMES_CANDIDATES, _FRAMES_CANDIDATES[0])
    )
    embeddings_root: Path = field(
        default_factory=lambda: _find_existing(_EMBEDDINGS_CANDIDATES, _EMBEDDINGS_CANDIDATES[0])
    )
    pair_index_path: Optional[Path] = None
    temporal_scores_path: Optional[Path] = None
    exclude_folders: list[str] = field(
        default_factory=lambda: ["reference for filtering", "reference images"]
    )
    eval_frames_root: Optional[Path] = Path("/media/HDD1/moritz/Extracted Frames/test")
    image_size: int = 224
    activity_alpha: float = 1.0  # exponent for activity-weighted sampling (0=uniform)
    retrieval_beta: float = 1.13  # min cosine sim for cross-video retrieval (0=no filter)


@dataclass
class DINOConfig:
    """DINO self-supervised training configuration."""

    backbone: str = "convnext_large"
    out_dim: int = 1536
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    n_global_crops: int = 2
    n_local_crops: int = 8
    global_crop_scale: tuple[float, float] = (0.4, 1.0)
    local_crop_scale: tuple[float, float] = (0.05, 0.4)
    temporal_neighbor_range: int = 2
    use_cross_video_pairs: bool = True
    cross_video_factor: float = 3.0


@dataclass
class VJEPAConfig:
    """V-JEPA self-supervised training configuration."""

    vjepa_backbone: str = "vit_large_patch16_224"
    predictor_depth: int = 6
    predictor_embed_dim: int = 512
    clip_length: int = 16
    clip_stride: int = 6
    mask_ratio: float = 0.75
    mask_tube_length: int = 2
    use_motion_guided_masking: bool = True
    motion_maps_root: Optional[Path] = None
    motion_bias_strength: float = 2.0
    patch_size: int = 16

    # Loss weights
    lambda_jepa: float = 1.0
    lambda_affinity: float = 0.5
    lambda_cross: float = 0.3
    lambda_sfdr: float = 0.1

    # Affinity distillation temperatures
    affinity_teacher_temp: float = 0.04
    affinity_student_temp: float = 0.1


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    base_lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    epochs: int = 30
    batch_size: int = 25
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 30
    student_temp: float = 0.1
    center_momentum: float = 0.9
    warmup_epochs: int = 10
    use_fp16: bool = True
    clip_grad: float = 3.0
    num_workers: int = 8
    save_freq: int = 10
    eval_freq: int = 1
    seed: int = 42
    device: str = "cuda"
    resume_from: Optional[Path] = None
    max_steps: Optional[int] = None
    knn_labels_root: Optional[Path] = None   # class-structured dir for k-NN eval
    evaluators: list[str] = field(default_factory=lambda: ["similarity"])
    use_wandb: bool = True
    wandb_project: str = "foundential-model"
    debug: bool = False


@dataclass
class Config(DataConfig, DINOConfig, VJEPAConfig, TrainConfig):
    """Combined config. EXPERIMENT_NAME drives all output paths."""

    EXPERIMENT_NAME: str = "baseline"
    ssl_method: str = "dino"  # "dino" or "vjepa" (see src/model/)

    # Derived paths — computed in __post_init__
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = PROJECT_ROOT / "output" / self.EXPERIMENT_NAME
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.eval_dir = self.output_dir / "eval"

    def save(self, path: Optional[Path] = None):
        """Serialize config to JSON."""
        path = path or (self.output_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        with open(path, "w") as f:
            json.dump(d, f, indent=2) 

    @classmethod
    def load(cls, path: Path) -> Config:
        """Load config from JSON."""
        with open(path) as f:
            d = json.load(f)
        # Convert path strings back
        for k in ("frames_root", "embeddings_root", "pair_index_path",
                   "temporal_scores_path", "motion_maps_root", "output_dir",
                   "checkpoint_dir", "eval_dir", "resume_from",
                   "eval_frames_root"):
            if k in d and d[k] is not None:
                d[k] = Path(d[k])
        # Remove derived fields (recomputed in __post_init__)
        for k in ("output_dir", "checkpoint_dir", "eval_dir"):
            d.pop(k, None)
        # Convert tuple fields
        for k in ("global_crop_scale", "local_crop_scale"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
