"""
End-task training entry point.

Two modes:
  - `--audit-only` (or no flags): summary of annotated frames across
    instruments + phases, restricted to the Foundational Model CSV.
  - `--experiment <name>` / `--config <path>`: load a config and dispatch
    to the task-specific trainer (e.g. YOLOv12 for instruments).

Usage:
    python -m src.end_tasks.train --audit-only
    python -m src.end_tasks.train --experiment instruments_yolov12
    python -m src.end_tasks.train --config output/end_tasks/instruments_yolov12/config.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Disable tqdm progress bars when stdout is not a TTY (e.g. nohup / log file).
# Must be set BEFORE importing ultralytics / tqdm so it takes effect.
if not sys.stdout.isatty():
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MININTERVAL", "60")

import pandas as pd

from .base import AUTOSAVE_DIR, INSTRUMENTS, PHASES, load_annotations
from .phases.dataloader import _match_name_to_autosave
from .instrument_tracking.dataloader import load_instrument_data
from .phases.dataloader import load_phase_data
from dataclasses import asdict

from .config import EndTaskConfig
from .experiments import END_TASK_EXPERIMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "src" / "data" / "foundational_model_videos.csv"


# ---------------------------------------------------------------------------
# Foundational model video audit
# ---------------------------------------------------------------------------

def _load_excel_video_names() -> dict[str, list[str]]:
    df = pd.read_csv(CSV_PATH)
    result: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        dataset = str(row.get("dataset", "unknown")).strip()
        if name and name != "nan":
            result[dataset].append(name)
    return dict(result)


_MRN_RE = re.compile(r"_(\d{7,9})[_/]")


def _match_excel_to_annotated(
    excel_names: dict[str, list[str]],
    annotated_videos: list[str],
) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = defaultdict(list)
    for sheet, names in excel_names.items():
        seen = set()
        for name in names:
            hit = _match_name_to_autosave(name, annotated_videos)
            if not hit:
                mrn_m = _MRN_RE.search(name)
                if mrn_m:
                    mrn = mrn_m.group(1)
                    for v in annotated_videos:
                        if mrn in v:
                            hit = v
                            break
            if hit and hit not in seen:
                matched[sheet].append(hit)
                seen.add(hit)
    return dict(matched)


def run_audit(tasks: list[str], require_all: bool):
    print(f"\n{'='*60}")
    print(f"  End-task data audit")
    print(f"  Tasks:       {', '.join(tasks)}")
    print(f"  Require all: {require_all}")
    print(f"{'='*60}\n")

    inst_data = load_instrument_data(AUTOSAVE_DIR) if "instruments" in tasks else {}
    phase_data = load_phase_data(AUTOSAVE_DIR) if "phases" in tasks else {}

    all_annotated = sorted(set(inst_data) | set(phase_data))

    excel_names = _load_excel_video_names()
    sheet_matches = _match_excel_to_annotated(excel_names, all_annotated)
    excel_videos = sorted({v for vlist in sheet_matches.values() for v in vlist})

    print(f"[excel] {sum(len(v) for v in excel_names.values())} rows across {len(excel_names)} sheets")
    print(f"[excel] {len(excel_videos)} matched to annotated videos\n")
    for sheet, matched in sheet_matches.items():
        total = len(excel_names.get(sheet, []))
        print(f"  {sheet:<35} {len(matched):>2} / {total} annotated")

    print()
    records = load_annotations(tasks=tasks, autosave_dir=AUTOSAVE_DIR, require_all=require_all)
    records = [r for r in records if r.video in set(excel_videos)]

    print(f"\n[dataset] {len(excel_videos)} videos, {len(records):,} frames total")

    if "instruments" in tasks:
        inst_frames = sum(1 for r in records if r.has_instruments)
        inst_counts = defaultdict(int)
        for r in records:
            for lbl in r.instrument_labels:
                inst_counts[lbl] += 1
        print(f"\n[instruments] {inst_frames:,} frames with instrument annotations")
        for lbl in INSTRUMENTS:
            if inst_counts[lbl]:
                print(f"  {lbl:<20} {inst_counts[lbl]:>6}")

    if "phases" in tasks:
        phase_frames = sum(1 for r in records if r.has_phase)
        phase_counts = defaultdict(int)
        for r in records:
            if r.has_phase:
                phase_counts[r.phase] += 1
        print(f"\n[phases] {phase_frames:,} frames with phase annotations")
        for ph in PHASES:
            if phase_counts[ph]:
                pct = 100 * phase_counts[ph] / phase_frames
                print(f"  {ph:<30} {phase_counts[ph]:>7} ({pct:.1f}%)")

    print(f"\n[done] Audit ready — {len(records):,} frames from {len(excel_videos)} videos")


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: EndTaskConfig, args) -> None:
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.base_lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.resume is not None:
        cfg.resume_from = args.resume
    if args.no_wandb:
        cfg.use_wandb = False
    if args.debug:
        cfg.debug = True


def _init_wandb(cfg: EndTaskConfig):
    """Start a wandb run before ultralytics does — lets us control name/config.

    Ultralytics' built-in wandb callback will reuse the active run rather than
    spawning a new one. Also flips the ultralytics setting on so its callback
    actually fires (it's off by default in recent versions).
    """
    if not cfg.use_wandb:
        return None
    try:
        import wandb
        flat_cfg = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        }
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.EXPERIMENT_NAME,
            config=flat_cfg,
            resume="allow",
            reinit="return_previous",
        )
        print(f"[wandb] Logging to {run.url}")
        try:
            from ultralytics import settings as _ult_settings
            if not _ult_settings.get("wandb", False):
                _ult_settings.update({"wandb": True})
                print("[wandb] enabled ultralytics wandb callback")
        except Exception as e:
            print(f"[wandb] could not enable ultralytics callback: {e}")
        return wandb
    except Exception as e:
        print(f"[wandb] Init failed: {e} — continuing without wandb")
        return None


def _finish_wandb(wb) -> None:
    if wb is None:
        return
    try:
        wb.finish()
    except Exception:
        pass


_OOM_HINTS = ("out of memory", "cuda out of memory", "cudnn_status_not_enough_workspace")


def _is_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _OOM_HINTS)


def _banner(text: str, bg: str) -> str:
    # bg: ANSI color code, e.g. "42" green, "41" red, "43" yellow
    line = "#" * (len(text) + 8)
    reset = "\033[0m"
    style = f"\033[{bg};97;1m"
    return f"\n{style}{line}{reset}\n{style}###  {text}  ###{reset}\n{style}{line}{reset}\n"


def _dispatch(cfg: EndTaskConfig) -> None:
    if cfg.task == "instruments":
        from .yolo.trainer import run
        run(cfg)
    elif cfg.task == "phases":
        raise NotImplementedError("phase training coming later")
    else:
        raise ValueError(f"unknown task: {cfg.task}")


def run_experiment(cfg: EndTaskConfig) -> None:
    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"  Task:       {cfg.task}")
    print(f"  Model:      {cfg.yolo_model}")
    print(f"  Epochs:     {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Splits CSV: {cfg.splits_csv}")
    print(f"{'='*60}\n")

    wb = _init_wandb(cfg)
    try:
        while cfg.batch_size >= 1:
            try:
                print(_banner(f"ATTEMPT: batch_size = {cfg.batch_size}", "44"))
                if wb is not None:
                    try:
                        wb.config.update({"batch_size": cfg.batch_size},
                                         allow_val_change=True)
                    except Exception:
                        pass
                _dispatch(cfg)
                print(_banner(f"TRAINING COMPLETE @ batch_size = {cfg.batch_size}", "42"))
                if wb is not None:
                    try:
                        wb.summary["final_batch_size"] = cfg.batch_size
                    except Exception:
                        pass
                return
            except Exception as e:
                if not _is_oom(e):
                    raise
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                new_bs = cfg.batch_size - 1
                print(_banner(f"OOM @ batch_size = {cfg.batch_size} -> RETRY {new_bs}", "41"))
                if new_bs < 1:
                    print(_banner("OOM at batch_size = 1 — cannot reduce further", "41"))
                    raise
                cfg.batch_size = new_bs
    finally:
        _finish_wandb(wb)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-task training entry point")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experiment", type=str, choices=list(END_TASK_EXPERIMENTS.keys()),
                       help="Named end-task experiment preset")
    group.add_argument("--config", type=Path,
                       help="Path to a saved end-task config.json")
    group.add_argument("--audit-only", action="store_true",
                       help="Print annotation summary and exit (no training)")
    group.add_argument("--all", action="store_true",
                       help="Run every experiment in END_TASK_EXPERIMENTS sequentially")

    parser.add_argument("--tasks", nargs="+", default=["instruments", "phases"],
                        choices=["instruments", "phases"],
                        help="Tasks to include in the audit (ignored during training)")
    parser.add_argument("--require-all", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.audit_only or (not args.experiment and not args.config and not args.all):
        run_audit(args.tasks, args.require_all)
        return

    if args.all:
        names = list(END_TASK_EXPERIMENTS.keys())
        print(f"[all] running {len(names)} experiments: {names}")
        failures = []
        for i, name in enumerate(names, 1):
            print(f"\n\033[44;97;1m [{i}/{len(names)}] {name} \033[0m")
            cfg = END_TASK_EXPERIMENTS[name]()
            _apply_overrides(cfg, args)
            if not cfg.splits_csv.is_file():
                print(f"[error] splits CSV not found at {cfg.splits_csv}")
                print(f"[error] run: python -m src.end_tasks.split")
                raise SystemExit(1)
            try:
                run_experiment(cfg)
            except Exception as e:
                print(f"\033[41;97;1m [FAIL] {name}: {e} \033[0m")
                failures.append((name, str(e)))
        if failures:
            print(f"\n[all] {len(failures)} failed: {[n for n,_ in failures]}")
            raise SystemExit(1)
        print(f"\n[all] all {len(names)} experiments completed")
        return

    if args.config:
        cfg = EndTaskConfig.load(args.config)
    else:
        cfg = END_TASK_EXPERIMENTS[args.experiment]()

    _apply_overrides(cfg, args)

    if not cfg.splits_csv.is_file():
        print(f"[error] splits CSV not found at {cfg.splits_csv}")
        print(f"[error] run: python -m src.end_tasks.split")
        raise SystemExit(1)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
