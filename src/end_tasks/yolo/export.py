"""
Materialize autosave instrument bboxes as a YOLO-format dataset tree.

Produces:
    <dataset_dir>/
        data.yaml
        images/{train,val,test}/   symlinks to /media/HDD1/.../frame_*.jpg
        labels/{train,val,test}/   one .txt per image (class cx cy w h, normalized)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..base import INSTRUMENTS, load_annotations
from ..config import EndTaskConfig
from ...data.dataset import discover_frames
from ..phases.dataloader import _match_name_to_autosave


def _resolve_video_to_frames(
    autosave_videos: list[str],
    frames_root: Path,
    exclude_folders: list[str],
) -> dict[str, list[Path]]:
    """Map each autosave stem to the sorted list of frame paths on disk."""
    discovered = discover_frames(frames_root, exclude_folders)
    # discovered keys are either "video" or "category/video" — build tail index
    tail_to_key = {k.rsplit("/", 1)[-1]: k for k in discovered}
    tails = list(tail_to_key.keys())

    out: dict[str, list[Path]] = {}
    for stem in autosave_videos:
        hit_tail = _match_name_to_autosave(stem, tails)
        if hit_tail is None:
            continue
        out[stem] = discovered[tail_to_key[hit_tail]]
    return out


_STANDARD_RES = [
    (640, 360), (704, 396), (1280, 720), (1600, 900),
    (1920, 1080), (2560, 1440), (3840, 2160),
]


def _annotation_size(records) -> tuple[int, int]:
    """Infer the image resolution the bboxes were drawn in, snapped to a standard.

    Autosave bboxes use the original video resolution, which may differ from
    the on-disk extracted frames. Since YOLO wants normalized coords, we only
    need the bbox-space W/H for normalization (not the disk image size).
    """
    max_x = 1
    max_y = 1
    for r in records:
        for x1, y1, x2, y2 in r.bboxes:
            if x2 > max_x:
                max_x = x2
            if y2 > max_y:
                max_y = y2
    for w, h in _STANDARD_RES:
        if max_x <= w and max_y <= h:
            return w, h
    return max_x, max_y


def _bbox_to_yolo(bbox: list[int], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = max(0, x2 - x1) / w
    bh = max(0, y2 - y1) / h
    return cx, cy, bw, bh


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def export_yolo_dataset(cfg: EndTaskConfig, split: dict[str, list[str]]) -> Path:
    """Build the YOLO dataset tree under cfg.dataset_dir and return the data.yaml path."""
    records = load_annotations(tasks=["instruments"], autosave_dir=cfg.autosave_dir)
    # Group by video for efficient per-video image dim lookup
    per_video: dict[str, list] = {}
    for r in records:
        if not r.has_instruments:
            continue
        per_video.setdefault(r.video, []).append(r)

    split_of: dict[str, str] = {}
    for s, vids in split.items():
        for v in vids:
            split_of[v] = s

    video_to_frames = _resolve_video_to_frames(
        list(per_video.keys()), cfg.frames_root, cfg.exclude_folders,
    )

    dataset_dir = cfg.dataset_dir
    for split_name in ("train", "val", "test"):
        (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    n_written = {"train": 0, "val": 0, "test": 0}
    n_skipped_no_frames = 0
    n_skipped_oob = 0

    n_skipped_empty = 0

    for video, recs in per_video.items():
        split_name = split_of.get(video)
        if split_name is None:
            continue
        frames = video_to_frames.get(video)
        if not frames:
            n_skipped_no_frames += len(recs)
            continue

        anno_w, anno_h = _annotation_size(recs)

        for r in recs:
            si = r.sample_idx
            if si >= len(frames):
                n_skipped_oob += 1
                continue

            lines = []
            for cls_id, box in zip(r.instrument_ids, r.bboxes):
                cx, cy, bw, bh = _bbox_to_yolo(box, anno_w, anno_h)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                n_skipped_empty += 1
                continue

            src_img = frames[si]
            stem = f"{video}__{si:06d}"
            dst_img = dataset_dir / "images" / split_name / f"{stem}{src_img.suffix}"
            dst_lbl = dataset_dir / "labels" / split_name / f"{stem}.txt"
            _symlink(src_img, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n")
            n_written[split_name] += 1

    print(f"[yolo-export] wrote frames: {n_written}")
    if n_skipped_no_frames:
        print(f"[yolo-export] skipped {n_skipped_no_frames} frames (no frames on disk for video)")
    if n_skipped_oob:
        print(f"[yolo-export] skipped {n_skipped_oob} frames (sample_idx out of range)")
    if n_skipped_empty:
        print(f"[yolo-export] skipped {n_skipped_empty} frames (all bboxes degenerate)")

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(INSTRUMENTS)},
    }, sort_keys=False))

    return data_yaml
