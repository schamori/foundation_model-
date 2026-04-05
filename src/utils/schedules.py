"""
Cosine schedules for LR, EMA momentum, weight decay, and teacher temperature.

All schedules follow the same pattern: cosine interpolation from start → end
over a given number of epochs, with optional linear warmup.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config


def cosine_schedule(base: float, end: float, epochs: int) -> np.ndarray:
    """Cosine annealing from base to end over epochs."""
    return end + 0.5 * (base - end) * (1 + np.cos(np.pi * np.arange(epochs) / epochs))


def lr_schedule(cfg: Config, steps_per_epoch: int) -> np.ndarray:
    """Per-step LR schedule: linear warmup → cosine decay to min_lr."""
    # Scale LR by batch size (linear scaling rule)
    scaled_lr = cfg.base_lr * cfg.batch_size / 256

    warmup_steps = min(cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch)
    total_steps = cfg.epochs * steps_per_epoch

    schedule = np.zeros(total_steps)

    # Linear warmup
    if warmup_steps > 0:
        schedule[:warmup_steps] = np.linspace(0, scaled_lr, warmup_steps)

    # Cosine decay
    remaining = total_steps - warmup_steps
    if remaining > 0:
        schedule[warmup_steps:] = cfg.min_lr + 0.5 * (scaled_lr - cfg.min_lr) * (
            1 + np.cos(np.pi * np.arange(remaining) / remaining)
        )

    return schedule


def ema_momentum_schedule(cfg: Config) -> np.ndarray:
    """Per-epoch EMA momentum: cosine from ema_momentum → ema_momentum_end."""
    return cosine_schedule(cfg.ema_momentum, cfg.ema_momentum_end, cfg.epochs)


def weight_decay_schedule(cfg: Config) -> np.ndarray:
    """Per-epoch weight decay: cosine from weight_decay → weight_decay_end."""
    return cosine_schedule(cfg.weight_decay, cfg.weight_decay_end, cfg.epochs)


def teacher_temp_schedule(cfg: Config) -> np.ndarray:
    """Per-epoch teacher temperature: linear warmup → constant."""
    schedule = np.ones(cfg.epochs) * cfg.teacher_temp
    if cfg.warmup_teacher_temp_epochs > 0 and cfg.epochs >= cfg.warmup_teacher_temp_epochs:
        warmup = np.linspace(
            cfg.warmup_teacher_temp, cfg.teacher_temp, cfg.warmup_teacher_temp_epochs
        )
        schedule[:cfg.warmup_teacher_temp_epochs] = warmup
    return schedule
