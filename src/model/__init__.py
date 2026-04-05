"""
Self-supervised learning methods.

Each method implements the SSLMethod interface so the Trainer can work
with any of them (DINO, V-JEPA, etc.) without changes.

Usage:
    method = get_ssl_method(cfg)
    method.build(cfg)
    for batch in loader:
        loss = method.train_step(batch, optimizer, scaler, epoch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import Config


class SSLMethod(ABC):
    """Interface for self-supervised pretraining methods."""

    name: str = "base"

    @abstractmethod
    def build(self, cfg: Config) -> None:
        """Initialize student, teacher, loss, and move to device."""
        ...

    @abstractmethod
    def train_step(
        self,
        batch: tuple,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> float:
        """Run one training step. Returns scalar loss."""
        ...

    @abstractmethod
    def update_teacher(self, momentum: float) -> None:
        """EMA update of teacher from student."""
        ...

    @abstractmethod
    def student_parameters(self) -> list[nn.Parameter]:
        """Parameters to optimize."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: Path, epoch: int, optimizer, scaler) -> None:
        """Save full training state."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path, optimizer, scaler) -> int:
        """Load training state. Returns start epoch."""
        ...

    @abstractmethod
    def collate_fn(self, batch: list) -> tuple:
        """Custom collate for DataLoader."""
        ...


# Registry
_SSL_METHODS: dict[str, type[SSLMethod]] = {}


def register_ssl_method(cls: type[SSLMethod]) -> type[SSLMethod]:
    """Decorator to register an SSL method."""
    _SSL_METHODS[cls.name] = cls
    return cls


def get_ssl_method(cfg: Config) -> SSLMethod:
    """Instantiate the SSL method specified in cfg.ssl_method."""
    if cfg.ssl_method not in _SSL_METHODS:
        available = ", ".join(_SSL_METHODS.keys())
        raise ValueError(f"Unknown ssl_method '{cfg.ssl_method}'. Available: {available}")
    return _SSL_METHODS[cfg.ssl_method]()


# Import methods to trigger registration
from . import dino, vjepa  # noqa: E402, F401
