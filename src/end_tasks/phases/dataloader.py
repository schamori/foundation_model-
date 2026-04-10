"""
Dataloader for surgical phase annotations.

Maps phase labels from phase-labels.json (via read_phases) to per-frame
phase indices, aligned with the autosave sample indices used by the
instrument tracking data.

Phase codes:
    0 = approach_and_exposure
    1 = treatment_phase
    2 = closure
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ..base import AUTOSAVE_DIR, PHASES, PHASE_TO_IDX


# ---------------------------------------------------------------------------
# Video name matching (prefix-based, same logic as object_tracking.py)
# ---------------------------------------------------------------------------

def _video_id(name: str) -> str:
    """Extract the short video identifier (e.g. '5ALA_003', 'MVD_004', 'RS-036').

    Uses the first two underscore-separated parts if the second part looks like
    a number/code (<=5 chars), otherwise just the part before the first
    underscore or hyphen-separated prefix.
    """
    parts = name.split("_", 2)
    if len(parts) >= 2 and len(parts[1]) <= 5:
        return f"{parts[0]}_{parts[1]}"
    # Fall back to prefix before first underscore (or whole name)
    idx = name.find("_")
    return name[:idx] if idx > 0 else name


def _match_name_to_autosave(
    name: str,
    autosave_dirs: list[str],
) -> str | None:
    """Match a phase-label video name to an autosave directory name.

    Tries: exact → video-id prefix match → contains.
    """
    name_lower = name.lower()
    vid_id = _video_id(name).lower()

    # Exact match
    for d in autosave_dirs:
        if d.lower() == name_lower:
            return d

    # Video-ID prefix match (5ALA_003 matches 5ALA_003_..., not 5ALA_006_...)
    for d in autosave_dirs:
        dl = d.lower()
        if dl == vid_id or dl.startswith(vid_id + "_") or dl.startswith(vid_id + "-"):
            return d

    # Substring match
    for d in autosave_dirs:
        if name_lower in d.lower():
            return d

    return None


# ---------------------------------------------------------------------------
# Phase → per-frame mapping (from evaluate.py:build_frame_phases)
# ---------------------------------------------------------------------------

def build_frame_phases(
    phases: list[dict],
    n_frames: int,
    phase_to_idx: dict[str, int] | None = None,
) -> np.ndarray:
    """Map phase timestamps to per-frame phase indices at 1 fps.

    Args:
        phases: list of {label, code, start_ms, end_ms}.
        n_frames: number of frames (sample indices 0..n_frames-1).
        phase_to_idx: code → index mapping. Defaults to PHASE_TO_IDX.

    Returns:
        int32 array of shape (n_frames,). -1 = no phase label.
    """
    if phase_to_idx is None:
        phase_to_idx = PHASE_TO_IDX

    frame_phases = np.full(n_frames, -1, dtype=np.int32)
    for phase in phases:
        start = int(phase["start_ms"] // 1000)
        end = int(phase["end_ms"] // 1000)
        idx = phase_to_idx.get(phase["code"], -1)
        if idx < 0:
            continue
        lo = max(0, start)
        hi = min(n_frames, end + 1)
        frame_phases[lo:hi] = idx
    return frame_phases


# ---------------------------------------------------------------------------
# Load phase data
# ---------------------------------------------------------------------------

def _get_n_frames_from_state(state_path: Path) -> int:
    """Get the number of frames from a state.json (max sample_idx + 1)."""
    with open(state_path) as f:
        data = json.load(f)
    bboxes = data.get("bboxes", {})
    if not bboxes:
        return 0
    return max(int(k) for k in bboxes.keys()) + 1


def _get_n_frames_from_phases(phases: list[dict]) -> int:
    """Estimate frame count from phase timestamps (max end_ms / 1000)."""
    if not phases:
        return 0
    max_end = max(p["end_ms"] for p in phases)
    return math.ceil(max_end / 1000) + 1


def load_phase_labels() -> list[dict]:
    """Load phase-labels.json and map video UUIDs to names via Excel.

    Returns list of {video_id, name, dataset, phases: [{label, code, start_ms, end_ms}]}.
    """
    from src.data.read_phases import map_phases_to_videos
    results, found, missing = map_phases_to_videos()
    return results


def load_phase_data(
    autosave_dir: Path = AUTOSAVE_DIR,
) -> dict[str, dict[int, dict]]:
    """Load phase annotations in the standardized format for base.py merging.

    Matches phase-labeled videos to autosave directories, then maps phase
    timestamps to per-frame indices.

    Returns:
        {autosave_video_name: {sample_idx: {"phase": str, "phase_id": int}}}
    """
    phase_data = load_phase_labels()

    # Get available autosave directories
    autosave_dirs = [
        d.name for d in sorted(autosave_dir.iterdir())
        if d.is_dir() and (d / "state.json").is_file()
    ]

    result: dict[str, dict[int, dict]] = {}

    for vinfo in phase_data:
        if vinfo["name"] is None or not vinfo["phases"]:
            continue

        # Match to autosave directory
        match = _match_name_to_autosave(vinfo["name"], autosave_dirs)
        if match is None:
            continue

        # Determine n_frames: prefer autosave state.json, fall back to phase timestamps
        state_path = autosave_dir / match / "state.json"
        if state_path.is_file():
            n_from_state = _get_n_frames_from_state(state_path)
        else:
            n_from_state = 0
        n_from_phases = _get_n_frames_from_phases(vinfo["phases"])
        n_frames = max(n_from_state, n_from_phases)

        if n_frames == 0:
            continue

        # Build per-frame phase array
        frame_phases = build_frame_phases(vinfo["phases"], n_frames)

        # Convert to standardized format (only labeled frames)
        frames: dict[int, dict] = {}
        for si in range(n_frames):
            pid = int(frame_phases[si])
            if pid < 0:
                continue
            frames[si] = {
                "phase": PHASES[pid],
                "phase_id": pid,
            }

        if frames:
            result[match] = frames

    return result


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def load_all_phase_annotations(
    autosave_dir: Path = AUTOSAVE_DIR,
) -> list[dict]:
    """Load phase annotations as a flat list of per-frame dicts.

    Convenience for analysis scripts. Each dict has:
        video, sample_idx, phase, phase_id.
    """
    data = load_phase_data(autosave_dir)
    records = []
    for video in sorted(data):
        for si in sorted(data[video]):
            d = data[video][si]
            records.append({
                "video": video,
                "sample_idx": si,
                "phase": d["phase"],
                "phase_id": d["phase_id"],
            })
    return records


if __name__ == "__main__":
    data = load_phase_data()
    total_frames = sum(len(frames) for frames in data.values())
    print(f"Loaded phase labels for {len(data)} videos, {total_frames} labeled frames")
    for video in sorted(data):
        frames = data[video]
        phases_seen = set(f["phase"] for f in frames.values())
        print(f"  {video}: {len(frames)} frames, phases: {sorted(phases_seen)}")
