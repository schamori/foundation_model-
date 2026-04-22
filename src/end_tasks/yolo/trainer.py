"""YOLOv12 instrument detection trainer — wraps ultralytics."""

from __future__ import annotations

import json
import os

from ..config import EndTaskConfig
from ..split import read_split_csv
from .export import export_yolo_dataset


def run(cfg: EndTaskConfig) -> None:
    if not cfg.splits_csv.is_file():
        raise FileNotFoundError(
            f"splits CSV not found at {cfg.splits_csv}. "
            f"Run: python -m src.end_tasks.split"
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()

    split = read_split_csv(cfg.splits_csv)
    data_yaml = export_yolo_dataset(cfg, split)

    from .sanity import sanity_check_augmented
    try:
        sanity_check_augmented(cfg, data_yaml, n=10)
    except Exception as e:
        print(f"[sanity] skipped: {e}")

    if cfg.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    from ultralytics import YOLO
    model = YOLO(cfg.yolo_model)
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        imgsz=cfg.image_size,
        device=cfg.device,
        workers=cfg.num_workers,
        project=str(cfg.output_dir),
        name="yolo",
        seed=cfg.seed,
        cos_lr=cfg.yolo_cos_lr,
        patience=cfg.yolo_patience,
        optimizer=cfg.yolo_optimizer,
        hsv_h=cfg.yolo_hsv_h,
        hsv_s=cfg.yolo_hsv_s,
        hsv_v=cfg.yolo_hsv_v,
        degrees=cfg.yolo_degrees,
        translate=cfg.yolo_translate,
        scale=cfg.yolo_scale,
        shear=cfg.yolo_shear,
        perspective=cfg.yolo_perspective,
        flipud=cfg.yolo_flipud,
        fliplr=cfg.yolo_fliplr,
        mosaic=cfg.yolo_mosaic,
        close_mosaic=cfg.yolo_close_mosaic,
        mixup=cfg.yolo_mixup,
        copy_paste=cfg.yolo_copy_paste,
        resume=bool(cfg.resume_from),
        exist_ok=True,
    )

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(cfg.output_dir),
        name="yolo_test",
        exist_ok=True,
    )
    (cfg.eval_dir / "test_metrics.json").write_text(
        json.dumps(metrics.results_dict, indent=2)
    )
    print(f"[yolo] test metrics → {cfg.eval_dir / 'test_metrics.json'}")
