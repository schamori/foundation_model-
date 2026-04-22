"""
Create stratified train/val/test split across annotated end-task videos.

Writes a CSV consumed by training — so splits are reproducible, auditable,
and hand-editable. Run once:

    python -m src.end_tasks.split [--seed 42] [--ratios 0.7 0.1 0.2]
                                  [--output src/data/end_task_splits.csv]

Stratification is video-level (no frame leakage) and balances, in order:
  1) surgery type (each type hits its own 70/10/20)
  2) instrument class distribution
  3) phase distribution

CSV schema:
    video, dataset, split, n_frames, n_instrument_frames, n_phase_frames
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .base import INSTRUMENTS, PHASES, load_annotations
from .config import PROJECT_ROOT
from .phases.dataloader import _match_name_to_autosave

DEFAULT_CSV = PROJECT_ROOT / "src" / "data" / "end_task_splits.csv"
DEFAULT_SOURCE_CSV = PROJECT_ROOT / "src" / "data" / "foundational_model_videos.csv"

_MRN_RE = re.compile(r"_(\d{7,9})[_/]")


# ---------------------------------------------------------------------------
# Per-video statistics
# ---------------------------------------------------------------------------

def _collect_video_stats(records) -> dict[str, dict]:
    """Aggregate FrameRecords into per-video class/frame counts."""
    stats: dict[str, dict] = {}
    for r in records:
        v = r.video
        s = stats.setdefault(v, {
            "n_frames": 0,
            "n_instrument_frames": 0,
            "n_phase_frames": 0,
            "instrument_counts": [0] * len(INSTRUMENTS),
            "phase_counts": [0] * len(PHASES),
        })
        s["n_frames"] += 1
        if r.has_instruments:
            s["n_instrument_frames"] += 1
            for iid in r.instrument_ids:
                if 0 <= iid < len(INSTRUMENTS):
                    s["instrument_counts"][iid] += 1
        if r.has_phase:
            s["n_phase_frames"] += 1
            s["phase_counts"][r.phase_id] += 1
    return stats


def _resolve_dataset_tags(videos: list[str], source_csv: Path) -> dict[str, str]:
    """Map each annotated video to its surgery type (dataset) from the CSV."""
    df = pd.read_csv(source_csv)
    video_to_dataset: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        dataset = str(row.get("dataset", "unknown")).strip()
        if not name or name == "nan":
            continue
        hit = _match_name_to_autosave(name, videos)
        if not hit:
            mrn_m = _MRN_RE.search(name)
            if mrn_m:
                mrn = mrn_m.group(1)
                for v in videos:
                    if mrn in v:
                        hit = v
                        break
        if hit and hit not in video_to_dataset:
            video_to_dataset[hit] = dataset
    return video_to_dataset


# ---------------------------------------------------------------------------
# Split targeting
# ---------------------------------------------------------------------------

def _target_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split counts for a group of size n, biased toward at-least-1-per-split.

    For n < 3: everything to train (can't divide meaningfully).
    For n >= 3: give each split 1, then distribute the remainder by ratio.
    Rounding residue is absorbed by train.
    """
    if n < 3:
        return (n, 0, 0)
    remaining = n - 3
    extras = [int(round(remaining * r)) for r in ratios]
    diff = remaining - sum(extras)
    extras[0] += diff
    return (1 + extras[0], 1 + extras[1], 1 + extras[2])


def _video_label_vec(video: str, stats: dict, dataset: str, datasets: list[str]) -> np.ndarray:
    """Concatenate one-hot dataset + normalized instrument + phase counts."""
    s = stats[video]
    ds_onehot = np.zeros(len(datasets), dtype=np.float32)
    ds_onehot[datasets.index(dataset)] = 1.0

    instr = np.asarray(s["instrument_counts"], dtype=np.float32)
    instr = instr / max(instr.sum(), 1.0)

    phase = np.asarray(s["phase_counts"], dtype=np.float32)
    phase = phase / max(phase.sum(), 1.0)

    return np.concatenate([ds_onehot, instr, phase])


# ---------------------------------------------------------------------------
# Group-level assignment
# ---------------------------------------------------------------------------

def _assign_group(
    videos: list[str],
    stats: dict,
    targets: tuple[int, int, int],
    seed: int,
) -> dict[str, str]:
    """Assign videos in one surgery group to splits.

    Prefers iterative-stratification on the (instrument + phase) frequency
    vector. Falls back to deterministic size-sorted zigzag if the package
    is missing or the group is tiny.
    """
    tr, va, te = targets
    slots = ["train"] * tr + ["val"] * va + ["test"] * te
    if len(slots) != len(videos):
        raise ValueError(f"slot count {len(slots)} != video count {len(videos)}")

    if len(videos) <= 3 or tr == len(videos):
        return _fallback_assign(videos, stats, slots, seed)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        return _fallback_assign(videos, stats, slots, seed)

    # Build per-video class presence (binary — iterstrat wants discrete labels)
    labels = []
    for v in videos:
        s = stats[v]
        row = [1 if c > 0 else 0 for c in s["instrument_counts"]]
        row += [1 if c > 0 else 0 for c in s["phase_counts"]]
        labels.append(row)
    y = np.asarray(labels)
    X = np.arange(len(videos)).reshape(-1, 1)

    try:
        # Stage 1: (val + test) vs train
        val_test_frac = (va + te) / len(videos)
        mss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_test_frac, random_state=seed
        )
        train_idx, valtest_idx = next(mss1.split(X, y))

        # Re-target to match the targets exactly (iterstrat is approximate)
        train_idx, valtest_idx = _rebalance(train_idx, valtest_idx, tr)

        # Stage 2: test vs val (within val_test)
        if va == 0 or te == 0:
            val_idx = valtest_idx[:va]
            test_idx = valtest_idx[va:]
        else:
            sub_y = y[valtest_idx]
            sub_X = np.arange(len(valtest_idx)).reshape(-1, 1)
            test_frac = te / (va + te)
            mss2 = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=test_frac, random_state=seed + 1
            )
            val_sub, test_sub = next(mss2.split(sub_X, sub_y))
            val_sub, test_sub = _rebalance(val_sub, test_sub, va)
            val_idx = valtest_idx[val_sub]
            test_idx = valtest_idx[test_sub]
    except Exception:
        return _fallback_assign(videos, stats, slots, seed)

    out = {}
    for i in train_idx:
        out[videos[i]] = "train"
    for i in val_idx:
        out[videos[i]] = "val"
    for i in test_idx:
        out[videos[i]] = "test"
    return out


def _rebalance(a_idx: np.ndarray, b_idx: np.ndarray, target_a: int) -> tuple[np.ndarray, np.ndarray]:
    """Move elements between a/b to make len(a) == target_a."""
    a_idx = list(a_idx)
    b_idx = list(b_idx)
    while len(a_idx) < target_a and b_idx:
        a_idx.append(b_idx.pop())
    while len(a_idx) > target_a:
        b_idx.append(a_idx.pop())
    return np.asarray(a_idx, dtype=int), np.asarray(b_idx, dtype=int)


def _fallback_assign(videos: list[str], stats: dict, slots: list[str], seed: int) -> dict[str, str]:
    """Size-sorted deterministic assignment when iterstrat isn't available."""
    rng = np.random.default_rng(seed)
    # Sort by total frames descending; shuffle ties within
    sort_key = [(-stats[v]["n_frames"], v) for v in videos]
    sort_key.sort()
    ordered = [v for _, v in sort_key]

    # Zigzag: largest → train, next → test, next → val, rotate
    rotation = ["train", "test", "val"]
    slots_by_rot = sorted(slots, key=lambda s: rotation.index(s) if s in rotation else 99)
    # Actually: distribute proportionally — pair each video with a slot,
    # placing large videos preferentially in the larger split.
    rng.shuffle(slots_by_rot)  # break deterministic ties softly with seed

    out = {}
    slot_iter = iter(slots)
    for v in ordered:
        out[v] = next(slot_iter)
    return out


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_split_csv(
    path: Path,
    videos_sorted: list[str],
    stats: dict,
    video_to_dataset: dict[str, str],
    assignments: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "dataset", "split",
                    "n_frames", "n_instrument_frames", "n_phase_frames"])
        for v in videos_sorted:
            s = stats[v]
            w.writerow([
                v,
                video_to_dataset.get(v, "unknown"),
                assignments[v],
                s["n_frames"],
                s["n_instrument_frames"],
                s["n_phase_frames"],
            ])


def read_split_csv(path: Path) -> dict[str, list[str]]:
    """Read split CSV and return {'train': [...], 'val': [...], 'test': [...]}."""
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in out:
                out[split].append(row["video"])
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    stats: dict,
    video_to_dataset: dict[str, str],
    assignments: dict[str, str],
) -> None:
    per_ds = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for v, ds in video_to_dataset.items():
        per_ds[ds][assignments[v]].append(v)

    print(f"\n{'='*72}")
    print(f"  {'dataset':<22} {'train':>14} {'val':>14} {'test':>14}")
    print(f"{'='*72}")
    totals = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    for ds in sorted(per_ds):
        row = f"  {ds:<22}"
        for split in ("train", "val", "test"):
            vids = per_ds[ds][split]
            nf = sum(stats[v]["n_frames"] for v in vids)
            totals[split][0] += len(vids)
            totals[split][1] += nf
            row += f" {len(vids):>3} vid {nf:>7}f"
        print(row)
    print(f"{'-'*72}")
    row = f"  {'TOTAL':<22}"
    for split in ("train", "val", "test"):
        row += f" {totals[split][0]:>3} vid {totals[split][1]:>7}f"
    print(row)
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build stratified end-task split CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV,
                        help="CSV with (dataset, name) rows to resolve surgery types")
    args = parser.parse_args()

    ratios = tuple(args.ratios)
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"ratios must sum to 1.0 (got {ratios})")

    print(f"[split] loading annotations...")
    records = load_annotations(tasks=["instruments", "phases"])
    stats = _collect_video_stats(records)
    videos = sorted(stats.keys())
    print(f"[split] {len(videos)} annotated videos, {sum(s['n_frames'] for s in stats.values()):,} frames")

    print(f"[split] resolving surgery types from {args.source_csv.name}...")
    video_to_dataset = _resolve_dataset_tags(videos, args.source_csv)

    # Filter to CSV-matched videos only (the Foundational Model target set)
    excluded = [v for v in videos if v not in video_to_dataset]
    if excluded:
        print(f"[split] excluding {len(excluded)} video(s) not in {args.source_csv.name}: {excluded}")
    videos = [v for v in videos if v in video_to_dataset]
    stats = {v: stats[v] for v in videos}
    print(f"[split] {len(videos)} videos in target set, "
          f"{sum(s['n_frames'] for s in stats.values()):,} frames")

    # Group by dataset
    groups: dict[str, list[str]] = defaultdict(list)
    for v, ds in video_to_dataset.items():
        groups[ds].append(v)

    assignments: dict[str, str] = {}
    for ds in sorted(groups):
        vids = sorted(groups[ds])
        n = len(vids)
        targets = _target_counts(n, ratios)
        if n < 3:
            warnings.warn(f"{ds}: only {n} video(s) — putting all in train")
        sub = _assign_group(vids, stats, targets, args.seed)
        assignments.update(sub)

    write_split_csv(args.output, videos, stats, video_to_dataset, assignments)
    print(f"[split] wrote {args.output}")

    _print_summary(stats, video_to_dataset, assignments)


if __name__ == "__main__":
    main()
