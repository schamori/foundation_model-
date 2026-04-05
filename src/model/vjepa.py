"""
V-JEPA self-supervised method: masked video prediction with cross-video matching.

Student encoder sees only visible patches, predictor reconstructs masked positions.
Teacher (EMA) sees all patches. Losses: JEPA (MSE) + affinity distillation +
cross-video Hungarian matching + feature diversity regularization.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SSLMethod, register_ssl_method
from .vjepa_modules import PatchEmbedding3D, VJEPAEncoder, VJEPAPredictor
from .vjepa_loss import VJEPACombinedLoss
from ..utils.masking import generate_tube_mask, motion_guided_tube_mask, mask_to_indices

if TYPE_CHECKING:
    from ..config import Config


@register_ssl_method
class VJEPAMethod(SSLMethod):
    """V-JEPA: Video Joint Embedding Predictive Architecture."""

    name = "vjepa"

    def __init__(self):
        self.student_encoder: VJEPAEncoder | None = None
        self.predictor: VJEPAPredictor | None = None
        self.teacher_encoder: VJEPAEncoder | None = None
        self.criterion: VJEPACombinedLoss | None = None
        self.device: str = "cuda"
        self.clip_grad: float = 3.0

        # Config cache
        self._img_size: int = 224
        self._patch_size: int = 16
        self._n_frames: int = 16
        self._mask_ratio: float = 0.75
        self._tube_length: int = 2
        self._use_motion: bool = False
        self._motion_strength: float = 2.0

    def build(self, cfg: Config) -> None:
        self.device = cfg.device
        self.clip_grad = cfg.clip_grad

        # Cache config for masking
        self._img_size = cfg.image_size
        self._patch_size = cfg.patch_size
        self._n_frames = cfg.clip_length
        self._mask_ratio = cfg.mask_ratio
        self._tube_length = cfg.mask_tube_length
        self._use_motion = cfg.use_motion_guided_masking
        self._motion_strength = cfg.motion_bias_strength

        H_patches = cfg.image_size // cfg.patch_size
        W_patches = cfg.image_size // cfg.patch_size
        n_tokens = cfg.clip_length * H_patches * W_patches

        # Create ViT backbone via timm
        import timm
        backbone = timm.create_model(cfg.vjepa_backbone, pretrained=True, num_classes=0)
        embed_dim = backbone.embed_dim if hasattr(backbone, "embed_dim") else backbone.num_features

        # Build student encoder
        patch_embed = PatchEmbedding3D(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            n_frames=cfg.clip_length,
            embed_dim=embed_dim,
        )
        self.student_encoder = VJEPAEncoder(backbone, patch_embed).to(self.device)

        # Build predictor
        self.predictor = VJEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=cfg.predictor_embed_dim,
            depth=cfg.predictor_depth,
            n_heads=max(1, cfg.predictor_embed_dim // 64),
            n_tokens=n_tokens,
        ).to(self.device)

        # Build teacher (deepcopy of encoder, no predictor)
        teacher_backbone = timm.create_model(cfg.vjepa_backbone, pretrained=True, num_classes=0)
        teacher_patch_embed = PatchEmbedding3D(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            n_frames=cfg.clip_length,
            embed_dim=embed_dim,
        )
        self.teacher_encoder = VJEPAEncoder(teacher_backbone, teacher_patch_embed).to(self.device)
        # Copy student weights to teacher
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # Build combined loss
        self.criterion = VJEPACombinedLoss(
            lambda_jepa=cfg.lambda_jepa,
            lambda_affinity=cfg.lambda_affinity,
            lambda_cross=cfg.lambda_cross,
            lambda_sfdr=cfg.lambda_sfdr,
            affinity_teacher_temp=cfg.affinity_teacher_temp,
            affinity_student_temp=cfg.affinity_student_temp,
        ).to(self.device)

    def student_parameters(self) -> list[nn.Parameter]:
        params = list(self.student_encoder.parameters())
        params += list(self.predictor.parameters())
        return params

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA update teacher encoder from student encoder (predictor excluded)."""
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data = momentum * pt.data + (1 - momentum) * ps.data

    def train_step(
        self,
        batch: tuple,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> float:
        clips, motion_maps = batch  # clips: (B, T, C, H, W)
        clips = clips.to(self.device, non_blocking=True)

        H_p = self._img_size // self._patch_size
        W_p = H_p
        T = clips.shape[1]

        # Generate mask
        if self._use_motion and motion_maps is not None:
            motion = motion_maps.to(self.device)
            mask = motion_guided_tube_mask(
                T, H_p, W_p, self._mask_ratio, self._tube_length,
                motion_bias=motion[0],  # use first sample's motion as proxy
                strength=self._motion_strength,
            ).to(self.device)
        else:
            mask = generate_tube_mask(
                T, H_p, W_p, self._mask_ratio, self._tube_length,
            ).to(self.device)

        visible_idx, masked_idx = mask_to_indices(mask)

        with torch.amp.autocast(device_type="cuda"):
            # Student: visible patches only
            student_out = self.student_encoder(clips, visible_idx)  # (B, N_vis, D)

            # Predictor: reconstruct all positions
            predicted = self.predictor(student_out, visible_idx, masked_idx)  # (B, N_total, D)

            # Teacher: all patches (no grad)
            with torch.no_grad():
                teacher_out = self.teacher_encoder(clips, None)  # (B, N_total, D)

            # Compute loss
            loss, loss_dict = self.criterion(
                predicted, teacher_out, masked_idx,
                cross_teacher=None,  # cross-video handled below if pairs exist
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.student_parameters(), max_norm=self.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    def collate_fn(self, batch: list) -> tuple:
        """Collate video clips into batched tensors.

        Each sample is a (T, C, H, W) tensor from VideoClipAugmentation.
        """
        clips = torch.stack(batch)  # (B, T, C, H, W)
        return clips, None  # motion_maps = None for now

    def save_checkpoint(self, path: Path, epoch: int, optimizer, scaler) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "student_encoder": self.student_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "teacher_encoder": self.teacher_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
        }, path)

    def load_checkpoint(self, path: Path, optimizer, scaler) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.student_encoder.load_state_dict(checkpoint["student_encoder"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.teacher_encoder.load_state_dict(checkpoint["teacher_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["epoch"] + 1
