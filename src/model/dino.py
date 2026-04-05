"""
DINO self-supervised method: student/teacher with EMA.

Implements SSLMethod so the Trainer can swap this for V-JEPA etc.
Absorbs logic from own experiments/training/DINO.py.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SSLMethod, register_ssl_method
from .backbone import create_backbone, get_embed_dim, pool_backbone_output

if TYPE_CHECKING:
    from ..config import Config


# ---------------------------------------------------------------------------
# DINO-specific modules
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Projection head with bottleneck + weight-normalized output layer."""

    def __init__(self, in_dim: int, out_dim: int = 1536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DINOLoss(nn.Module):
    """Cross-entropy loss with teacher centering and temperature sharpening."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04,
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor,
                update_center: bool = True) -> torch.Tensor:
        student_out = student_out / self.student_temp
        teacher_out = F.softmax(
            (teacher_out - self.center) / self.teacher_temp, dim=-1
        ).detach()
        loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        if update_center:
            self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> None:
        self.center = (self.center * self.center_momentum
                       + teacher_out.mean(0, keepdim=True) * (1 - self.center_momentum))


class _DINOModel(nn.Module):
    """Backbone + projection head (used for both student and teacher)."""

    def __init__(self, backbone: nn.Module, embed_dim: int,
                 out_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(embed_dim, out_dim, hidden_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = pool_backbone_output(self.backbone, x)
        return self.head(features)


# ---------------------------------------------------------------------------
# DINOMethod — the SSLMethod implementation
# ---------------------------------------------------------------------------

@register_ssl_method
class DINOMethod(SSLMethod):
    """DINO self-supervised learning with student/teacher + EMA."""

    name = "dino"

    def __init__(self):
        self.student: _DINOModel | None = None
        self.teacher: _DINOModel | None = None
        self.criterion: DINOLoss | None = None
        self.device: str = "cuda"
        self.n_global: int = 2
        self.n_local: int = 8
        self.clip_grad: float = 3.0

    def build(self, cfg: Config) -> None:
        self.device = cfg.device
        self.n_global = cfg.n_global_crops
        self.n_local = cfg.n_local_crops
        self.clip_grad = cfg.clip_grad

        backbone = create_backbone(cfg)
        embed_dim = get_embed_dim(backbone)

        self.student = _DINOModel(
            backbone, embed_dim, cfg.out_dim, cfg.hidden_dim, cfg.bottleneck_dim
        ).to(self.device)

        self.teacher = copy.deepcopy(self.student).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DINOLoss(
            cfg.out_dim, cfg.teacher_temp, cfg.student_temp, cfg.center_momentum
        ).to(self.device)

    def student_parameters(self) -> list[nn.Parameter]:
        return list(self.student.parameters())

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = momentum * pt.data + (1 - momentum) * ps.data

    def train_step(
        self,
        batch: tuple,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> float:
        globals_, locals_ = batch
        global_batch = torch.cat(globals_, dim=0).to(self.device, non_blocking=True)
        local_batch = torch.cat(locals_, dim=0).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            # Teacher sees only global crops
            with torch.no_grad():
                teacher_out = self.teacher(global_batch)

            # Student sees all crops (split by resolution)
            student_global = self.student(global_batch)
            student_local = self.student(local_batch)
            student_out = torch.cat([student_global, student_local], dim=0)

            # Reshape for vectorized cross-crop loss
            B = globals_[0].shape[0]
            n_crops = self.n_global + self.n_local
            teacher_out = teacher_out.view(self.n_global, B, -1)
            student_out = student_out.view(n_crops, B, -1)

            # Update center with flattened teacher output
            self.criterion._update_center(teacher_out.flatten(0, 1))

            # Cross-crop masking: skip where student crop == teacher crop
            s_grid = torch.arange(n_crops, device=self.device).unsqueeze(1).expand(-1, self.n_global)
            t_grid = torch.arange(self.n_global, device=self.device).unsqueeze(0).expand(n_crops, -1)
            mask = s_grid != t_grid

            s_batch = student_out[s_grid[mask]].flatten(0, 1)
            t_batch = teacher_out[t_grid[mask]].flatten(0, 1)

            loss = self.criterion(s_batch, t_batch, update_center=False)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=self.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    def collate_fn(self, batch: list) -> tuple:
        """Collate multi-crop samples into batched tensor lists."""
        globals_ = [torch.stack([s[0][i] for s in batch]) for i in range(self.n_global)]
        locals_ = [torch.stack([s[1][i] for s in batch]) for i in range(self.n_local)]
        return globals_, locals_

    def save_checkpoint(self, path: Path, epoch: int, optimizer, scaler) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "center": self.criterion.center,
            "epoch": epoch,
        }, path)

    def load_checkpoint(self, path: Path, optimizer, scaler) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(checkpoint["student"])
        self.teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        self.criterion.center = checkpoint["center"].to(self.device)
        return checkpoint["epoch"] + 1
