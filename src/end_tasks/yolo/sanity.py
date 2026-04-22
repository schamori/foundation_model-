"""
Pre-training sanity check: materialize 10 random samples from the ultralytics
augmentation pipeline and overlay YOLO bboxes on them.

The images saved here are *exactly* what the model will see after mosaic,
copy-paste, mixup, HSV jitter, rotation/translate/scale, flips, letterbox.
A quick eyeball of ``output/end_tasks/<exp>/sanity_check/`` before each run
catches misconfigured augmentations, broken labels, and resolution issues
that normally only surface as bad val metrics many hours in.

Uses ultralytics' own ``YOLODataset`` so we don't reimplement its transforms
— we just sample from it.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from ..base import INSTRUMENTS
from ..config import EndTaskConfig

# BGR palette for drawing — distinct for each instrument class.
_PALETTE_BGR = [
    (132, 99, 255), (235, 162, 54), (86, 205, 255),
    (192, 192, 75), (255, 102, 153), (64, 159, 255), (180, 180, 180),
]


def _draw_boxes(img_bgr: np.ndarray, bboxes_xywhn, cls_ids, names: list[str]) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for (cx, cy, bw, bh), c in zip(bboxes_xywhn, cls_ids):
        x1 = int(round((cx - bw / 2) * W)); y1 = int(round((cy - bh / 2) * H))
        x2 = int(round((cx + bw / 2) * W)); y2 = int(round((cy + bh / 2) * H))
        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
        col = _PALETTE_BGR[int(c) % len(_PALETTE_BGR)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = names[int(c)] if int(c) < len(names) else str(int(c))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ly = max(th + 2, y1)
        cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly), col, -1)
        cv2.putText(out, label, (x1 + 2, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def sanity_check_augmented(
    cfg: EndTaskConfig,
    data_yaml: Path,
    n: int = 10,
    split: str = "train",
) -> Path:
    """Render ``n`` random augmented samples and save them + a 2×5 contact sheet.

    Returns the sanity-check directory.
    """
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    out_dir = cfg.output_dir / "sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = check_det_dataset(str(data_yaml))

    overrides = dict(
        hsv_h=cfg.yolo_hsv_h, hsv_s=cfg.yolo_hsv_s, hsv_v=cfg.yolo_hsv_v,
        degrees=cfg.yolo_degrees, translate=cfg.yolo_translate,
        scale=cfg.yolo_scale, shear=cfg.yolo_shear,
        perspective=cfg.yolo_perspective,
        flipud=cfg.yolo_flipud, fliplr=cfg.yolo_fliplr,
        mosaic=cfg.yolo_mosaic, mixup=cfg.yolo_mixup,
        copy_paste=cfg.yolo_copy_paste,
        imgsz=cfg.image_size,
    )
    hyp = get_cfg(DEFAULT_CFG, overrides=overrides)

    ds = YOLODataset(
        img_path=data[split],
        imgsz=cfg.image_size,
        augment=(split == "train"),
        hyp=hyp,
        rect=False,
        batch_size=1,
        stride=32,
        pad=0.0,
        data=data,
        task="detect",
        single_cls=False,
    )

    if len(ds) == 0:
        print("[sanity] empty dataset — skipping")
        return out_dir

    n = min(n, len(ds))
    rng = random.Random(cfg.seed)
    idxs = rng.sample(range(len(ds)), n)

    raw_names = data.get("names", {i: n_ for i, n_ in enumerate(INSTRUMENTS)})
    if isinstance(raw_names, dict):
        names = [raw_names[i] for i in sorted(raw_names)]
    else:
        names = list(raw_names)

    saved = []
    for k, i in enumerate(idxs):
        sample = ds[i]

        img_t = sample["img"]
        if isinstance(img_t, torch.Tensor):
            arr = img_t.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW → HWC
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                # ultralytics returns uint8 by default for detection
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(img_t)

        # Ultralytics uses RGB internally; convert for cv2 write
        img_bgr = arr[:, :, ::-1].copy() if arr.shape[-1] == 3 else arr

        bboxes = sample.get("bboxes", None)
        cls = sample.get("cls", None)
        box_list, cls_list = [], []
        if bboxes is not None and len(bboxes) > 0:
            b = bboxes.detach().cpu().numpy() if hasattr(bboxes, "detach") else np.asarray(bboxes)
            c = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
            box_list = b.tolist()
            cls_list = c.astype(int).flatten().tolist()

        drawn = _draw_boxes(img_bgr, box_list, cls_list, names)

        # Footer strip with metadata
        h, w = drawn.shape[:2]
        footer_h = 20
        out = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
        out[:h] = drawn
        txt = f"idx={i}  boxes={len(cls_list)}  size={w}x{h}  aug=hsv({cfg.yolo_hsv_h},{cfg.yolo_hsv_s},{cfg.yolo_hsv_v}) mos={cfg.yolo_mosaic} mix={cfg.yolo_mixup} cp={cfg.yolo_copy_paste}"
        cv2.putText(out, txt, (4, h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)

        path = out_dir / f"sanity_{k:02d}.jpg"
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(out)

    if saved:
        # Contact sheet: 2 rows × 5 cols (or close, depending on n)
        cols = min(5, len(saved))
        rows = (len(saved) + cols - 1) // cols
        th, tw = saved[0].shape[:2]
        sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for k, im in enumerate(saved):
            r, c = k // cols, k % cols
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"[sanity] wrote {len(saved)} augmented samples → {out_dir}")
    return out_dir
