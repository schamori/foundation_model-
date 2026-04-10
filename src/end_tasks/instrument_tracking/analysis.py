"""
Instrument tracking dataset statistics.

Prints overall and per-video breakdowns of instrument occurrences,
co-occurrence patterns, and coverage stats using the tracking annotations.

Usage:
    python -m src.end_tasks.instrument_tracking.analysis
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ..base import AUTOSAVE_DIR, INSTRUMENTS
from .dataloader import load_all_annotations, load_video_annotations


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def overall_stats(autosave_dir: Path = AUTOSAVE_DIR):
    anns = load_all_annotations(autosave_dir)
    videos = set(a.video for a in anns)

    print_section("DATASET OVERVIEW")
    print(f"  Videos with annotations:  {len(videos)}")
    print(f"  Total annotated frames:   {len(anns)}")
    frames_with_instruments = sum(1 for a in anns if a.bboxes)
    print(f"  Frames with instruments:  {frames_with_instruments}")
    total_boxes = sum(len(a.bboxes) for a in anns)
    print(f"  Total bounding boxes:     {total_boxes}")
    if frames_with_instruments:
        print(f"  Avg instruments/frame:    {total_boxes / frames_with_instruments:.2f}")

    # --- Per-instrument counts ---
    print_section("PER-INSTRUMENT COUNTS")
    inst_frames = Counter()  # instrument -> n frames where it appears
    inst_boxes = Counter()   # instrument -> total bbox count
    for a in anns:
        seen = set()
        for label in a.labels:
            inst_boxes[label] += 1
            if label not in seen:
                inst_frames[label] += 1
                seen.add(label)

    header = f"  {'Instrument':<16} {'Boxes':>8} {'Frames':>8} {'% frames':>9}"
    print(header)
    print(f"  {'-'*43}")
    for inst in INSTRUMENTS:
        boxes = inst_boxes.get(inst, 0)
        frames = inst_frames.get(inst, 0)
        pct = 100 * frames / len(anns) if anns else 0
        print(f"  {inst:<16} {boxes:>8} {frames:>8} {pct:>8.1f}%")
    print(f"  {'-'*43}")
    print(f"  {'TOTAL':<16} {total_boxes:>8}")

    # --- Co-occurrence ---
    print_section("CO-OCCURRENCE (how often 2 instruments appear in the same frame)")
    cooccur = Counter()
    for a in anns:
        unique = sorted(set(a.labels))
        for i, l1 in enumerate(unique):
            for l2 in unique[i + 1:]:
                cooccur[(l1, l2)] += 1

    top_pairs = cooccur.most_common(15)
    if top_pairs:
        print(f"  {'Pair':<36} {'Frames':>8}")
        print(f"  {'-'*46}")
        for (l1, l2), count in top_pairs:
            print(f"  {l1 + ' + ' + l2:<36} {count:>8}")

    # --- Instruments per frame distribution ---
    print_section("INSTRUMENTS PER FRAME DISTRIBUTION")
    n_per_frame = Counter(len(set(a.labels)) for a in anns)
    for n in sorted(n_per_frame):
        count = n_per_frame[n]
        pct = 100 * count / len(anns) if anns else 0
        bar = "#" * int(pct / 2)
        print(f"  {n} instruments: {count:>6} frames ({pct:>5.1f}%) {bar}")

    # --- Per-video summary ---
    print_section("PER-VIDEO SUMMARY")
    video_anns = defaultdict(list)
    for a in anns:
        video_anns[a.video].append(a)

    print(f"  {'Video':<50} {'Frames':>7} {'Boxes':>7} {'Instruments':>12}")
    print(f"  {'-'*78}")
    for video in sorted(video_anns):
        v_anns = video_anns[video]
        v_boxes = sum(len(a.bboxes) for a in v_anns)
        v_instruments = sorted(set(l for a in v_anns for l in a.labels))
        short = video[:48] if len(video) > 48 else video
        print(f"  {short:<50} {len(v_anns):>7} {v_boxes:>7} {', '.join(v_instruments)}")

    # --- Per-video instrument frame counts ---
    print_section("PER-VIDEO INSTRUMENT FRAME COUNTS")
    active_instruments = [inst for inst in INSTRUMENTS if inst_boxes.get(inst, 0) > 0]
    col_w = 8
    header = f"  {'Video':<30}" + "".join(f"{inst[:col_w]:>{col_w}}" for inst in active_instruments)
    print(header)
    print(f"  {'-' * (30 + col_w * len(active_instruments))}")
    for video in sorted(video_anns):
        v_anns = video_anns[video]
        counts = Counter()
        for a in v_anns:
            for label in set(a.labels):
                counts[label] += 1
        short = video[:28] if len(video) > 28 else video
        row = f"  {short:<30}"
        for inst in active_instruments:
            c = counts.get(inst, 0)
            row += f"{c if c else '.':>{col_w}}"
        print(row)


if __name__ == "__main__":
    overall_stats()
