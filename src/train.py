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
    transform = build_augmentation(cfg)

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

    sampler = dataset.sampler()

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=method.collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def run_experiment(cfg: Config) -> None:
    """Run a single experiment."""
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"  Method:     {cfg.ssl_method}")
    print(f"  Backbone:   {cfg.backbone if cfg.ssl_method == 'dino' else cfg.vjepa_backbone}")
    print(f"  Epochs:     {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"{'=' * 60}\n")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Build method
    method = get_ssl_method(cfg)
    method.build(cfg)

    # Build dataloader
    loader = build_dataloader(cfg, method)

    # Train
    trainer = Trainer(method, loader, cfg)
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

    args = parser.parse_args()

    if args.config:
        cfg = Config.load(args.config)
        _apply_overrides(cfg, args)
        run_experiment(cfg)
    elif args.experiment:
        cfg = EXPERIMENTS[args.experiment]()
        _apply_overrides(cfg, args)
        run_experiment(cfg)
    else:
        # No arguments → run all experiments from get_experiment_configs()
        configs = get_experiment_configs()
        print(f"Running {len(configs)} experiments")
        for i, cfg in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] {cfg.EXPERIMENT_NAME}")
            _apply_overrides(cfg, args)
            run_experiment(cfg)


if __name__ == "__main__":
    main()
