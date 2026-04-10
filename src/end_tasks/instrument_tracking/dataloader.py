"""
Dataloader for instrument tracking annotations.

Loads tracking labels produced by the SAM2 labeling tool
(object_tracking.py) from tracking_exports/autosave/.

Each sample is one frame with its set of instrument bounding boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ..base import AUTOSAVE_DIR, INSTRUMENTS, INSTRUMENT_TO_IDX


@dataclass
class FrameAnnotation:
    """All instrument annotations for a single frame."""
    video: str
    sample_idx: int
    bboxes: list[list[int]]          # [[x1, y1, x2, y2], ...]
    labels: list[str]                # instrument name per bbox
    label_ids: list[int]             # INSTRUMENT_TO_IDX per bbox
    oof_oids: set[int] = field(default_factory=set)  # object IDs that are OOF


def _parse_oof(oof_data: dict | list) -> dict[int, list[list]]:
    """Parse OOF data into {oid: [[start, end], ...]} handling all formats."""
    result = {}
    if isinstance(oof_data, list):
        # Old format: [[frame, oid], ...]
        for si, oid in oof_data:
            result[int(oid)] = [[int(si), None]]
    elif isinstance(oof_data, dict):
        for oid_str, val in oof_data.items():
            oid = int(oid_str)
            if isinstance(val, list) and val and isinstance(val[0], list):
                result[oid] = val  # new ranges format
            elif isinstance(val, (int, float)):
                result[oid] = [[int(val), None]]  # old single-value
    return result


def _is_oof(oof_ranges: dict[int, list[list]], oid: int, sample_idx: int) -> bool:
    """Check if object is out-of-frame at a given sample index."""
    if oid not in oof_ranges:
        return False
    for start, end in oof_ranges[oid]:
        if sample_idx >= start and (end is None or sample_idx <= end):
            return True
    return False


def load_video_annotations(state_path: Path) -> list[FrameAnnotation]:
    """Load all frame annotations from a single video's state.json."""
    video_name = state_path.parent.name
    with open(state_path) as f:
        data = json.load(f)

    objects = data.get("objects", {})
    bboxes_data = data.get("bboxes", {})
    oof_data = _parse_oof(data.get("oof", {}))

    if not objects or not bboxes_data:
        return []

    annotations = []
    for si_str, frame_bboxes in sorted(bboxes_data.items(), key=lambda x: int(x[0])):
        si = int(si_str)
        bboxes = []
        labels = []
        label_ids = []
        oof_oids = set()

        for oid_str, bbox in frame_bboxes.items():
            oid = int(oid_str)
            if bbox is None:
                continue
            if oid_str not in objects:
                continue
            if _is_oof(oof_data, oid, si):
                oof_oids.add(oid)
                continue

            label = objects[oid_str]["label"]
            bboxes.append(bbox)
            labels.append(label)
            label_ids.append(INSTRUMENT_TO_IDX.get(label, len(INSTRUMENTS) - 1))

        annotations.append(FrameAnnotation(
            video=video_name,
            sample_idx=si,
            bboxes=bboxes,
            labels=labels,
            label_ids=label_ids,
            oof_oids=oof_oids,
        ))

    return annotations


def load_all_annotations(
    autosave_dir: Path = AUTOSAVE_DIR,
    min_objects: int = 0,
) -> list[FrameAnnotation]:
    """Load annotations from all videos in the autosave directory."""
    all_annotations = []
    for video_dir in sorted(autosave_dir.iterdir()):
        state = video_dir / "state.json"
        if not state.is_file():
            continue
        anns = load_video_annotations(state)
        if min_objects > 0:
            anns = [a for a in anns if len(a.bboxes) >= min_objects]
        all_annotations.extend(anns)
    return all_annotations


# ---------------------------------------------------------------------------
# Standardized format for base.py merging
# ---------------------------------------------------------------------------

def load_instrument_data(
    autosave_dir: Path = AUTOSAVE_DIR,
) -> dict[str, dict[int, dict]]:
    """Load instrument annotations in the standardized format.

    Returns:
        {video_name: {sample_idx: {bboxes, instrument_labels, instrument_ids, oof_oids}}}
    """
    result: dict[str, dict[int, dict]] = {}
    for video_dir in sorted(autosave_dir.iterdir()):
        state = video_dir / "state.json"
        if not state.is_file():
            continue
        anns = load_video_annotations(state)
        if not anns:
            continue
        frames: dict[int, dict] = {}
        for a in anns:
            frames[a.sample_idx] = {
                "bboxes": a.bboxes,
                "instrument_labels": a.labels,
                "instrument_ids": a.label_ids,
                "oof_oids": a.oof_oids,
            }
        result[anns[0].video] = frames
    return result


# ---------------------------------------------------------------------------
# Standalone Dataset (kept for backward compatibility)
# ---------------------------------------------------------------------------

class InstrumentTrackingDataset(Dataset):
    """PyTorch dataset over instrument tracking annotations."""

    def __init__(
        self,
        autosave_dir: Path = AUTOSAVE_DIR,
        min_objects: int = 1,
        videos: Optional[list[str]] = None,
    ):
        self.annotations = load_all_annotations(autosave_dir, min_objects)
        if videos is not None:
            video_set = set(videos)
            self.annotations = [a for a in self.annotations if a.video in video_set]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        a = self.annotations[idx]
        bboxes = np.array(a.bboxes, dtype=np.int32) if a.bboxes else np.zeros((0, 4), dtype=np.int32)
        label_ids = np.array(a.label_ids, dtype=np.int64)
        return {
            "video": a.video,
            "sample_idx": a.sample_idx,
            "bboxes": bboxes,
            "label_ids": label_ids,
            "labels": a.labels,
        }


if __name__ == "__main__":
    anns = load_all_annotations()
    print(f"Loaded {len(anns)} annotated frames from {len(set(a.video for a in anns))} videos")
    ds = InstrumentTrackingDataset(min_objects=1)
    print(f"Dataset: {len(ds)} frames with >= 1 visible instrument")
    if len(ds):
        sample = ds[0]
        print(f"Sample: video={sample['video']}, frame={sample['sample_idx']}, "
              f"bboxes={sample['bboxes'].shape}, labels={sample['labels']}")
