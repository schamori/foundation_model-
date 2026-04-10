"""
Shared infrastructure for end-task evaluation datasets.

Provides a unified FrameRecord that can carry instrument bboxes, phase labels,
or both.  Each sub-module (instrument_tracking, phases) has its own loader that
returns ``dict[str, dict[int, dict]]`` (video → sample_idx → fields).
This module merges them and wraps the result in a PyTorch Dataset.

Usage:
    from src.end_tasks.base import load_annotations, EndTaskDataset

    # Load only instruments
    records = load_annotations(tasks=["instruments"])

    # Load only phases
    records = load_annotations(tasks=["phases"])

    # Combined — keeps frames present in ANY task
    records = load_annotations(tasks=["instruments", "phases"])

    # As a Dataset (drops frames with no visible data)
    ds = EndTaskDataset(tasks=["instruments", "phases"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOSAVE_DIR = PROJECT_ROOT / "tracking_exports" / "autosave"

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

INSTRUMENTS = [
    "Bipolar", "Microdissectors", "Suction", "Drills",
    "Microscissors", "CUSA", "Others",
]
INSTRUMENT_TO_IDX = {name: i for i, name in enumerate(INSTRUMENTS)}

PHASES = [
    "approach_and_exposure",
    "treatment_phase",
    "closure",
]
PHASE_TO_IDX = {name: i for i, name in enumerate(PHASES)}


# ---------------------------------------------------------------------------
# Unified per-frame record
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """All annotations for a single frame, across tasks."""
    video: str
    sample_idx: int

    # -- instrument tracking (empty when not loaded / no instruments) --
    bboxes: list[list[int]] = field(default_factory=list)
    instrument_labels: list[str] = field(default_factory=list)
    instrument_ids: list[int] = field(default_factory=list)
    oof_oids: set[int] = field(default_factory=set)

    # -- phase recognition (-1 / None when not loaded / unlabeled) --
    phase: str | None = None
    phase_id: int = -1

    @property
    def has_instruments(self) -> bool:
        return len(self.bboxes) > 0

    @property
    def has_phase(self) -> bool:
        return self.phase_id >= 0


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _merge(
    instrument_data: dict[str, dict[int, dict]] | None,
    phase_data: dict[str, dict[int, dict]] | None,
) -> list[FrameRecord]:
    """Merge per-frame dicts from different loaders into FrameRecords.

    Keeps the *union* of all (video, sample_idx) pairs — missing fields
    stay at their defaults.
    """
    # Collect all (video, sample_idx) pairs
    keys: dict[str, set[int]] = {}
    for source in (instrument_data, phase_data):
        if source is None:
            continue
        for video, frames in source.items():
            keys.setdefault(video, set()).update(frames.keys())

    records: list[FrameRecord] = []
    for video in sorted(keys):
        for si in sorted(keys[video]):
            rec = FrameRecord(video=video, sample_idx=si)

            if instrument_data and video in instrument_data:
                inst = instrument_data[video].get(si)
                if inst:
                    rec.bboxes = inst["bboxes"]
                    rec.instrument_labels = inst["instrument_labels"]
                    rec.instrument_ids = inst["instrument_ids"]
                    rec.oof_oids = inst.get("oof_oids", set())

            if phase_data and video in phase_data:
                ph = phase_data[video].get(si)
                if ph:
                    rec.phase = ph["phase"]
                    rec.phase_id = ph["phase_id"]

            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_annotations(
    tasks: list[str] | None = None,
    autosave_dir: Path = AUTOSAVE_DIR,
    require_all: bool = False,
) -> list[FrameRecord]:
    """Load annotations for the requested tasks and merge them.

    Args:
        tasks: list of task names to load. Supported: "instruments", "phases".
               None means all tasks.
        autosave_dir: root autosave directory for instrument tracking.
        require_all: if True, only keep frames that have data from ALL
                     requested tasks (intersection). Default is union.
    """
    if tasks is None:
        tasks = ["instruments", "phases"]

    instrument_data = None
    phase_data = None

    if "instruments" in tasks:
        from .instrument_tracking.dataloader import load_instrument_data
        instrument_data = load_instrument_data(autosave_dir)

    if "phases" in tasks:
        from .phases.dataloader import load_phase_data
        phase_data = load_phase_data(autosave_dir)

    records = _merge(instrument_data, phase_data)

    if require_all and len(tasks) > 1:
        checks = []
        if "instruments" in tasks:
            checks.append(lambda r: r.has_instruments)
        if "phases" in tasks:
            checks.append(lambda r: r.has_phase)
        records = [r for r in records if all(c(r) for c in checks)]

    return records


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class EndTaskDataset(Dataset):
    """PyTorch dataset over merged end-task annotations.

    Each item returns a dict with:
        - video (str)
        - sample_idx (int)
        - bboxes (np.ndarray, N×4 int32)       — if instruments loaded
        - instrument_ids (np.ndarray, N int64)  — if instruments loaded
        - instrument_labels (list[str])         — if instruments loaded
        - phase (str | None)                    — if phases loaded
        - phase_id (int)                        — if phases loaded
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        autosave_dir: Path = AUTOSAVE_DIR,
        require_all: bool = False,
        min_instruments: int = 0,
        videos: Optional[list[str]] = None,
    ):
        records = load_annotations(tasks, autosave_dir, require_all)

        if min_instruments > 0:
            records = [r for r in records if len(r.bboxes) >= min_instruments]
        if videos is not None:
            vset = set(videos)
            records = [r for r in records if r.video in vset]

        self.records = records
        self.tasks = tasks or ["instruments", "phases"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        out = {
            "video": r.video,
            "sample_idx": r.sample_idx,
        }
        if "instruments" in self.tasks:
            out["bboxes"] = (
                np.array(r.bboxes, dtype=np.int32)
                if r.bboxes
                else np.zeros((0, 4), dtype=np.int32)
            )
            out["instrument_ids"] = np.array(r.instrument_ids, dtype=np.int64)
            out["instrument_labels"] = r.instrument_labels
        if "phases" in self.tasks:
            out["phase"] = r.phase
            out["phase_id"] = r.phase_id
        return out
