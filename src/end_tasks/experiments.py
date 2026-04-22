"""End-task experiment presets."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EndTaskConfig


@dataclass
class InstrumentYOLOv12(EndTaskConfig):
    EXPERIMENT_NAME: str = "instruments_yolov12"
    task: str = "instruments"
    yolo_model: str = "yolov12s.pt"
    epochs: int = 100
    batch_size: int = 120


@dataclass
class InstrumentYOLOv12_Geometric(EndTaskConfig):
    """All geometric augs on, zero color/brightness jitter."""
    EXPERIMENT_NAME: str = "instruments_yolov12_geometric"
    task: str = "instruments"
    yolo_model: str = "yolov12s.pt"
    epochs: int = 100
    batch_size: int = 120

    # No color
    yolo_hsv_h: float = 0.0
    yolo_hsv_s: float = 0.0
    yolo_hsv_v: float = 0.0

    # Full geometric
    yolo_degrees: float = 5.0
    yolo_translate: float = 0.1
    yolo_scale: float = 0.5
    yolo_shear: float = 2.0
    yolo_perspective: float = 0.0005
    yolo_flipud: float = 0.0
    yolo_fliplr: float = 0.5

    yolo_mosaic: float = 1.0
    yolo_close_mosaic: int = 10
    yolo_mixup: float = 0.0
    yolo_copy_paste: float = 0.0


@dataclass
class InstrumentYOLOv12_AllAugs(EndTaskConfig):
    """Everything on — color + geometric + mosaic + mixup + built-in copy-paste."""
    EXPERIMENT_NAME: str = "instruments_yolov12_allaugs"
    task: str = "instruments"
    yolo_model: str = "yolov12s.pt"
    epochs: int = 100
    batch_size: int = 120

    yolo_hsv_h: float = 0.015
    yolo_hsv_s: float = 0.7
    yolo_hsv_v: float = 0.4

    yolo_degrees: float = 5.0
    yolo_translate: float = 0.1
    yolo_scale: float = 0.5
    yolo_shear: float = 2.0
    yolo_perspective: float = 0.0005
    yolo_flipud: float = 0.0
    yolo_fliplr: float = 0.5

    yolo_mosaic: float = 1.0
    yolo_close_mosaic: int = 10
    yolo_mixup: float = 0.1
    yolo_copy_paste: float = 0.1


END_TASK_EXPERIMENTS: dict[str, type[EndTaskConfig]] = {
    "instruments_yolov12": InstrumentYOLOv12,
    "instruments_yolov12_geometric": InstrumentYOLOv12_Geometric,
    "instruments_yolov12_allaugs": InstrumentYOLOv12_AllAugs,
}


def get_end_task_configs() -> list[EndTaskConfig]:
    return [cls() for cls in END_TASK_EXPERIMENTS.values()]
