"""
Main training entry point for foundation model pretraining.

Supports both DINO and V-JEPA via the SSLMethod registry. Wires together:
config → dataset → augmentation → method → trainer.

Usage:
    python -m src.train --experiment full
    python -m src.train --experiment vjepa_full
    python -m src.train --config output/my_experiment/config.json
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .experiments import (
    Baseline, WithTemporal, WithCrossVideo, Full,
    VJEPABaseline, VJEPAWithMotion, VJEPAWithCrossVideo, VJEPAFull,
    get_experiment_configs,
)
from .model import get_ssl_method
from .data.dataset import SurgicalFrameDataset
from .utils.augmentations import build_augmentation
from .utils.training import Trainer


EXPERIMENTS = {
    "baseline": Baseline,
    "with_temporal": WithTemporal,
    "with_crossvideo": WithCrossVideo,
    "full": Full,
    "vjepa_baseline": VJEPABaseline,
    "vjepa_motion": VJEPAWithMotion,
    "vjepa_crossvideo": VJEPAWithCrossVideo,
    "vjepa_full": VJEPAFull,
}


def build_dataloader(cfg: Config, method) -> DataLoader:
    """Build dataset and dataloader for the configured method."""
    debug = cfg.debug
    if debug:
        print("[debug] Building augmentation...", flush=True)
    transform = build_augmentation(cfg)

    if debug:
        print(f"[debug] Building dataset (frames_root={cfg.frames_root})...", flush=True)
    dataset = SurgicalFrameDataset(
        frames_root=cfg.frames_root,
        exclude_folders=cfg.exclude_folders,
        temporal_scores_path=cfg.temporal_scores_path,
        pair_index_path=cfg.pair_index_path,
        temporal_neighbor_range=cfg.temporal_neighbor_range,
        use_cross_video_pairs=cfg.use_cross_video_pairs,
        activity_alpha=cfg.activity_alpha,
        transform=transform,
        mode=cfg.ssl_method,
        clip_length=cfg.clip_length,
        clip_stride=cfg.clip_stride,
    )

    if debug:
        print(f"[debug] Building sampler ({len(dataset)} samples)...", flush=True)
    sampler = dataset.sampler()

    if debug:
        print(f"[debug] Creating DataLoader (batch_size={cfg.batch_size}, "
              f"num_workers={cfg.num_workers}, pin_memory=True)...", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=method.collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=3 if cfg.num_workers > 0 else None,
    )
    if debug:
        print(f"[debug] DataLoader ready ({len(loader)} steps/epoch)", flush=True)
    return loader


def run_experiment(cfg: Config) -> None:
    """Run a single experiment."""
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"  Method:     {cfg.ssl_method}")
    print(f"  Backbone:   {cfg.backbone if cfg.ssl_method == 'dino' else cfg.vjepa_backbone}")
    print(f"  Epochs:     {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"{'=' * 60}\n")

    debug = cfg.debug

    # Set seed
    torch.manual_seed(cfg.seed)

    # Build method
    if debug:
        print("[debug] Building SSL method...", flush=True)
    method = get_ssl_method(cfg)
    if debug:
        print("[debug] Calling method.build()...", flush=True)
    method.build(cfg)
    if debug:
        print("[debug] Method ready", flush=True)

    # Build dataloader
    loader = build_dataloader(cfg, method)

    # Train
    if debug:
        print("[debug] Creating Trainer...", flush=True)
    trainer = Trainer(method, loader, cfg)
    if debug:
        print("[debug] Starting training loop...", flush=True)
    trainer.train()


def _apply_overrides(cfg: Config, args) -> None:
    """Apply CLI overrides to a config."""
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.base_lr = args.lr
    if args.resume is not None:
        cfg.resume_from = args.resume
    if args.device is not None:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.eval_frames_root is not None:
        cfg.eval_frames_root = args.eval_frames_root
    if args.frames_root is not None:
        cfg.frames_root = Path(args.frames_root)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.evaluators is not None:
        cfg.evaluators = args.evaluators
    if hasattr(args, "debug") and args.debug:
        cfg.debug = True


def _run_one(cfg: Config) -> None:
    """Run a single experiment in a subprocess."""
    try:
        run_experiment(cfg)
    except Exception as e:
        print(f"[FAILED] {cfg.EXPERIMENT_NAME} on {cfg.device}: {e}")


def _run_gpu_queue(cfgs: list[Config]) -> None:
    """Run a list of experiments sequentially (used as subprocess target)."""
    for cfg in cfgs:
        _run_one(cfg)


def _run_parallel(configs: list[Config], gpus: list[int]) -> None:
    """Distribute experiments round-robin across GPUs and run in parallel."""
    # Assign devices round-robin
    for i, cfg in enumerate(configs):
        cfg.device = f"cuda:{gpus[i % len(gpus)]}"

    print(f"Running {len(configs)} experiments across GPUs {gpus}")
    for cfg in configs:
        print(f"  {cfg.EXPERIMENT_NAME} → {cfg.device}")

    # Group by GPU — run experiments on same GPU sequentially
    from collections import defaultdict
    gpu_groups: dict[str, list[Config]] = defaultdict(list)
    for cfg in configs:
        gpu_groups[cfg.device].append(cfg)

    ctx = mp.get_context("spawn")
    processes = []
    for device, cfgs in gpu_groups.items():
        p = ctx.Process(target=_run_gpu_queue, args=(cfgs,))
        p.start()
        processes.append((device, p))

    for device, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[WARNING] Process for {device} exited with code {p.exitcode}")

    print("All experiments finished.")


def main():
    parser = argparse.ArgumentParser(description="Foundation model pretraining")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment", type=str, choices=list(EXPERIMENTS.keys()),
        help="Named experiment preset",
    )
    group.add_argument(
        "--config", type=Path,
        help="Path to config.json from a previous run",
    )

    # Override common params
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop each epoch after this many steps (for testing)")
    parser.add_argument("--eval-frames-root", type=Path, default=None,
                        help="Validation frames directory for similarity eval")
    parser.add_argument("--frames-root", type=str, default=None,
                        help="Training frames directory")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (0 = main process, avoids shm issues)")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU ids to use. Multiple experiments run in parallel across GPUs "
                             "(e.g. --gpus 0 1 assigns experiment 0→cuda:0, experiment 1→cuda:1)")
    parser.add_argument("--evaluators", type=str, nargs="+", default=None,
                        help="Which evaluators to run (e.g. similarity cross_video_retrieval knn)")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose logging for debugging hangs/startup issues")

    args = parser.parse_args()

    if args.config:
        cfg = Config.load(args.config)
        _apply_overrides(cfg, args)
        run_experiment(cfg)
    elif args.experiment:
        cfg = EXPERIMENTS[args.experiment]()
        _apply_overrides(cfg, args)
        if args.gpus and not args.device:
            cfg.device = f"cuda:{args.gpus[0]}"
        run_experiment(cfg)
    else:
        # No arguments → run all experiments from get_experiment_configs()
        configs = get_experiment_configs()
        for cfg in configs:
            _apply_overrides(cfg, args)

        if args.gpus and len(args.gpus) > 1 and not args.device:
            _run_parallel(configs, args.gpus)
        else:
            if args.gpus and not args.device:
                for cfg in configs:
                    cfg.device = f"cuda:{args.gpus[0]}"
            print(f"Running {len(configs)} experiments sequentially")
            for i, cfg in enumerate(configs, 1):
                print(f"\n[{i}/{len(configs)}] {cfg.EXPERIMENT_NAME}")
                run_experiment(cfg)


if __name__ == "__main__":
    main()
