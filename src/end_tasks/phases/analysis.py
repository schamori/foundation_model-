"""
Phase annotation statistics.

Prints overall and per-video breakdowns of surgical phase distributions
using the phase annotations mapped to autosave videos.

Usage:
    python -m src.end_tasks.phases.analysis
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ..base import AUTOSAVE_DIR, PHASES
from .dataloader import load_phase_data


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def overall_stats(autosave_dir: Path = AUTOSAVE_DIR):
    data = load_phase_data(autosave_dir)

    if not data:
        print("No phase-labeled videos found.")
        return

    total_frames = sum(len(frames) for frames in data.values())

    print_section("PHASE DATASET OVERVIEW")
    print(f"  Videos with phase labels:  {len(data)}")
    print(f"  Total labeled frames:      {total_frames}")

    # --- Per-phase counts ---
    print_section("PER-PHASE COUNTS")
    phase_counts = Counter()
    for frames in data.values():
        for f in frames.values():
            phase_counts[f["phase"]] += 1

    header = f"  {'Phase':<28} {'Frames':>8} {'% frames':>9}"
    print(header)
    print(f"  {'-' * 47}")
    for phase in PHASES:
        count = phase_counts.get(phase, 0)
        pct = 100 * count / total_frames if total_frames else 0
        bar = "#" * int(pct / 2)
        print(f"  {phase:<28} {count:>8} {pct:>8.1f}%  {bar}")
    print(f"  {'-' * 47}")
    print(f"  {'TOTAL':<28} {total_frames:>8}")

    # --- Phase transitions ---
    print_section("PHASE TRANSITIONS PER VIDEO")
    print(f"  {'Video':<40} {'Transitions':>12} {'Phases seen':>12}")
    print(f"  {'-' * 66}")
    for video in sorted(data):
        frames = data[video]
        sorted_sis = sorted(frames.keys())
        phases_seq = [frames[si]["phase"] for si in sorted_sis]
        transitions = sum(1 for i in range(1, len(phases_seq)) if phases_seq[i] != phases_seq[i - 1])
        phases_seen = len(set(phases_seq))
        short = video[:38] if len(video) > 38 else video
        print(f"  {short:<40} {transitions:>12} {phases_seen:>12}")

    # --- Per-video phase distribution ---
    print_section("PER-VIDEO PHASE DISTRIBUTION")
    col_w = max(len(p) for p in PHASES) + 2
    header = f"  {'Video':<30}" + "".join(f"{p:>{col_w}}" for p in PHASES)
    print(header)
    print(f"  {'-' * (30 + col_w * len(PHASES))}")
    for video in sorted(data):
        frames = data[video]
        counts = Counter(f["phase"] for f in frames.values())
        short = video[:28] if len(video) > 28 else video
        row = f"  {short:<30}"
        for phase in PHASES:
            c = counts.get(phase, 0)
            row += f"{c if c else '.':>{col_w}}"
        print(row)

    # --- Per-video phase coverage (% of video frames that are labeled) ---
    print_section("PHASE COVERAGE")
    print(f"  {'Video':<40} {'Labeled':>8} {'Duration(s)':>12} {'Coverage':>9}")
    print(f"  {'-' * 71}")
    for video in sorted(data):
        frames = data[video]
        n_labeled = len(frames)
        if frames:
            max_si = max(frames.keys())
            min_si = min(frames.keys())
            duration = max_si - min_si + 1
        else:
            duration = 0
        pct = 100 * n_labeled / duration if duration > 0 else 0
        short = video[:38] if len(video) > 38 else video
        print(f"  {short:<40} {n_labeled:>8} {duration:>12} {pct:>8.1f}%")


if __name__ == "__main__":
    overall_stats()
