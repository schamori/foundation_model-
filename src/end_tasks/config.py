"""
End-task training config.

Mirrors src/config.py but lives in a separate namespace so SSL pretraining
and end-task training don't collide. EXPERIMENT_NAME drives the output tree:

    output/end_tasks/<EXPERIMENT_NAME>/config.json
    output/end_tasks/<EXPERIMENT_NAME>/checkpoints/
    output/end_tasks/<EXPERIMENT_NAME>/eval/
    output/end_tasks/<EXPERIMENT_NAME>/yolo_dataset/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_FRAMES_CANDIDATES = [
    Path("/media/HDD1/moritz/Extracted Frames"),
    Path("/media/HDD1/moritz/Extracted Frames/train"),
    Path("/media/HDD1/moritz/Extracted Frames/test"),
]


def _find_existing(candidates: list[Path], fallback: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return fallback


@dataclass
class EndTaskDataConfig:
    frames_root: Path = field(
        default_factory=lambda: _find_existing(_FRAMES_CANDIDATES, _FRAMES_CANDIDATES[0])
    )
    autosave_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "tracking_exports" / "autosave"
    )
    csv_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "src" / "data" / "foundational_model_videos.csv"
    )
    exclude_folders: list[str] = field(
        default_factory=lambda: [
            "reference for filtering", "reference images",
            "anti_references", "train", "test",
        ]
    )
    image_size: int = 640


@dataclass
class EndTaskTrainConfig:
    epochs: int = 100
    batch_size: int = 150
    base_lr: float = 1e-3
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 42
    resume_from: Optional[Path] = None
    use_wandb: bool = True
    wandb_project: str = "foundential-model-endtask"
    debug: bool = False


@dataclass
class YOLOConfig:
    yolo_model: str = "yolov12s.pt"
    yolo_optimizer: str = "auto"
    yolo_patience: int = 30
    yolo_cos_lr: bool = True

    # Color / brightness
    yolo_hsv_h: float = 0.015
    yolo_hsv_s: float = 0.7
    yolo_hsv_v: float = 0.4

    # Geometric
    yolo_degrees: float = 0.0
    yolo_translate: float = 0.1
    yolo_scale: float = 0.5
    yolo_shear: float = 0.0
    yolo_perspective: float = 0.0
    yolo_flipud: float = 0.0
    yolo_fliplr: float = 0.5

    # Mosaic / mixup / built-in copy-paste
    yolo_mosaic: float = 1.0
    yolo_close_mosaic: int = 10
    yolo_mixup: float = 0.0
    yolo_copy_paste: float = 0.0


@dataclass
class EndTaskConfig(EndTaskDataConfig, EndTaskTrainConfig, YOLOConfig):
    EXPERIMENT_NAME: str = "instruments_yolov12"
    task: str = "instruments"

    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
    split_seed: int = 42
    splits_csv: Path = field(
        default_factory=lambda: PROJECT_ROOT / "src" / "data" / "end_task_splits.csv"
    )

    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)
    dataset_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = PROJECT_ROOT / "output" / "end_tasks" / self.EXPERIMENT_NAME
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.eval_dir = self.output_dir / "eval"
        self.dataset_dir = self.output_dir / "yolo_dataset"

    def save(self, path: Optional[Path] = None):
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
    def load(cls, path: Path) -> EndTaskConfig:
        with open(path) as f:
            d = json.load(f)
        for k in ("frames_root", "autosave_dir", "csv_path", "splits_csv",
                 "resume_from", "output_dir", "checkpoint_dir", "eval_dir",
                 "dataset_dir"):
            if k in d and d[k] is not None:
                d[k] = Path(d[k])
        for k in ("output_dir", "checkpoint_dir", "eval_dir", "dataset_dir"):
            d.pop(k, None)
        if "split_ratios" in d:
            d["split_ratios"] = tuple(d["split_ratios"])
        return cls(**d)
