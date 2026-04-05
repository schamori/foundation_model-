"""
Generic Trainer that works with any SSLMethod.

Handles the training loop, schedule application, checkpointing, and evaluation.
The SSL-specific logic (forward, loss, teacher update) is delegated to the method.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .schedules import lr_schedule, ema_momentum_schedule, weight_decay_schedule, teacher_temp_schedule

if TYPE_CHECKING:
    from ..config import Config
    from ..model import SSLMethod


class Trainer:
    """SSL-agnostic training loop."""

    def __init__(self, method: SSLMethod, dataloader: DataLoader, cfg: Config):
        self.method = method
        self.loader = dataloader
        self.cfg = cfg

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            method.student_parameters(),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(enabled=cfg.use_fp16)

        # Schedules
        steps_per_epoch = len(dataloader)
        self.lr_sched = lr_schedule(cfg, steps_per_epoch)
        self.ema_sched = ema_momentum_schedule(cfg)
        self.wd_sched = weight_decay_schedule(cfg)
        self.temp_sched = teacher_temp_schedule(cfg)

        self.start_epoch = 0

        # Resume
        if cfg.resume_from and cfg.resume_from.exists():
            print(f"[trainer] Resuming from {cfg.resume_from}")
            self.start_epoch = method.load_checkpoint(cfg.resume_from, self.optimizer, self.scaler)
            print(f"[trainer] Resumed at epoch {self.start_epoch}")

    def train(self) -> None:
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.save()

        steps_per_epoch = len(self.loader)

        for epoch in range(self.start_epoch, cfg.epochs):
            self._apply_epoch_schedules(epoch)

            total_loss = 0.0
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", disable=not sys.stdout.isatty())

            for step, batch in enumerate(pbar):
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break

                global_step = epoch * steps_per_epoch + step

                # Per-step LR
                lr = float(self.lr_sched[min(global_step, len(self.lr_sched) - 1)])
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                loss = self.method.train_step(batch, self.optimizer, self.scaler, epoch)

                # EMA teacher update
                momentum = float(self.ema_sched[epoch])
                self.method.update_teacher(momentum)

                total_loss += loss
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

            avg_loss = total_loss / steps_per_epoch
            print(f"[trainer] Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

            # Checkpoint
            if (epoch + 1) % cfg.save_freq == 0 or epoch == cfg.epochs - 1:
                ckpt_path = cfg.checkpoint_dir / f"{self.method.name}_epoch{epoch + 1}.pt"
                self.method.save_checkpoint(ckpt_path, epoch, self.optimizer, self.scaler)
                print(f"[trainer] Saved checkpoint → {ckpt_path}")

            # Evaluation
            if (epoch + 1) % cfg.eval_freq == 0:
                self._run_eval(epoch + 1)

    def _apply_epoch_schedules(self, epoch: int) -> None:
        """Apply per-epoch weight decay and teacher temperature."""
        wd = float(self.wd_sched[epoch])
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = wd

        # Update teacher temperature if the method supports it
        if hasattr(self.method, "criterion") and hasattr(self.method.criterion, "teacher_temp"):
            self.method.criterion.teacher_temp = float(self.temp_sched[epoch])

    def _run_eval(self, epoch: int) -> None:
        """Run registered evaluators on the latest checkpoint."""
        ckpt_path = self.cfg.checkpoint_dir / f"{self.method.name}_epoch{epoch}.pt"
        if not ckpt_path.exists():
            return
        try:
            from ..evaluation import run_all_evaluations
            metrics = run_all_evaluations(self.cfg, ckpt_path, epoch)
            print(f"[trainer] Eval epoch {epoch}: {metrics}")
        except Exception as e:
            print(f"[trainer] Eval failed: {e}")
