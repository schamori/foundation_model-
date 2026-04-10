"""
Generic Trainer that works with any SSLMethod.

Handles the training loop, schedule application, checkpointing, and evaluation.
The SSL-specific logic (forward, loss, teacher update) is delegated to the method.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import sys
import time
import torch


def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable h/m/s string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
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
        self.debug = cfg.debug

        # Optimizer
        if self.debug:
            print("[debug] Creating optimizer...", flush=True)
        self.optimizer = torch.optim.AdamW(
            method.student_parameters(),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(enabled=cfg.use_fp16)

        # Schedules
        if self.debug:
            print("[debug] Building LR/EMA/WD/temp schedules...", flush=True)
        steps_per_epoch = len(dataloader)
        self.lr_sched = lr_schedule(cfg, steps_per_epoch)
        self.ema_sched = ema_momentum_schedule(cfg)
        self.wd_sched = weight_decay_schedule(cfg)
        self.temp_sched = teacher_temp_schedule(cfg)
        if self.debug:
            print("[debug] Trainer init complete", flush=True)

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

        is_tty = sys.stdout.isatty()
        log_interval = max(1, steps_per_epoch // 5)  # log ~5 times per epoch

        for epoch in range(self.start_epoch, cfg.epochs):
            self._apply_epoch_schedules(epoch)
            epoch_t0 = time.time()

            if not is_tty:
                print(f"[trainer] Epoch {epoch + 1}/{cfg.epochs} starting "
                      f"({steps_per_epoch} steps)", flush=True)

            total_loss = 0.0
            if self.debug:
                print(f"[debug] Creating DataLoader iterator...", flush=True)
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", disable=not is_tty)

            for step, batch in enumerate(pbar):
                if self.debug and step == 0:
                    print(f"[debug] First batch loaded successfully "
                          f"(keys={list(batch.keys()) if isinstance(batch, dict) else type(batch).__name__})",
                          flush=True)
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break

                global_step = epoch * steps_per_epoch + step

                # Per-step LR
                lr = float(self.lr_sched[min(global_step, len(self.lr_sched) - 1)])
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                if self.debug and step == 0:
                    print(f"[debug] Running first train_step...", flush=True)
                loss = self.method.train_step(batch, self.optimizer, self.scaler, epoch)
                if self.debug and step == 0:
                    print(f"[debug] First step done, loss={loss:.4f}", flush=True)

                # EMA teacher update
                momentum = float(self.ema_sched[epoch])
                self.method.update_teacher(momentum)

                total_loss += loss
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

                if not is_tty and (step + 1) % log_interval == 0:
                    elapsed = time.time() - epoch_t0
                    eta = elapsed / (step + 1) * (steps_per_epoch - step - 1)
                    avg_so_far = total_loss / (step + 1)
                    print(f"[trainer]   step {step + 1}/{steps_per_epoch} — "
                          f"loss: {loss:.4f} (avg: {avg_so_far:.4f}), "
                          f"lr: {lr:.2e}, "
                          f"elapsed: {_fmt_time(elapsed)}, eta: {_fmt_time(eta)}", flush=True)

            epoch_time = time.time() - epoch_t0
            actual_steps = min(step + 1, cfg.max_steps) if cfg.max_steps else steps_per_epoch
            avg_loss = total_loss / actual_steps
            remaining_epochs = cfg.epochs - epoch - 1
            remaining_time = epoch_time * remaining_epochs
            print(f"[trainer] Epoch {epoch + 1} — avg loss: {avg_loss:.4f}, "
                  f"time: {_fmt_time(epoch_time)}, "
                  f"est. remaining: {_fmt_time(remaining_time)}")

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
