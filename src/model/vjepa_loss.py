"""
V-JEPA loss functions.

Four losses combined:
- JEPALoss: MSE masked prediction (L_jepa)
- AffinityLoss: KL on pairwise similarity matrices (L_affinity)
- CrossMatchingLoss: Hungarian-matched cross-video alignment (L_cross)
- SFDRLoss: Stochastic Feature Diversity Regularizer (L_sfdr)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """MSE between predicted and target representations at masked positions."""

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        masked_indices: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, N_total, D) from predictor
            target: (B, N_total, D) from teacher encoder
            masked_indices: (N_masked,) positions to compute loss over
        """
        B = predicted.shape[0]
        idx = masked_indices.unsqueeze(0).expand(B, -1)  # (B, N_masked)
        idx = idx.unsqueeze(-1).expand(-1, -1, predicted.shape[-1])

        pred_masked = torch.gather(predicted, 1, idx)   # (B, N_masked, D)
        target_masked = torch.gather(target, 1, idx)     # (B, N_masked, D)

        return F.mse_loss(pred_masked, target_masked)


class AffinityLoss(nn.Module):
    """KL divergence between student and teacher pairwise affinity matrices.

    Distills the relational structure (which tokens are similar to each other)
    from teacher to student.
    """

    def __init__(self, teacher_temp: float = 0.04, student_temp: float = 0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_features: (B, N, D) from predictor
            teacher_features: (B, N, D) from teacher encoder
        """
        # Pairwise cosine similarity matrices
        s_norm = F.normalize(student_features, dim=-1)
        t_norm = F.normalize(teacher_features, dim=-1)

        A_student = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, N, N)
        A_teacher = torch.bmm(t_norm, t_norm.transpose(1, 2))  # (B, N, N)

        # Row-wise softmax with temperatures
        p_teacher = F.softmax(A_teacher / self.teacher_temp, dim=-1).detach()
        log_p_student = F.log_softmax(A_student / self.student_temp, dim=-1)

        # KL divergence per row, averaged
        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        return kl


class CrossMatchingLoss(nn.Module):
    """Cross-video Hungarian matching loss.

    Finds optimal one-to-one token correspondence between student predictions
    (Patient A) and teacher representations (Patient B) using the Hungarian
    algorithm, then computes a differentiable loss on matched pairs.

    The assignment is detached (no gradient through the discrete optimization).
    """

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_features: (B, N, D) predictor output from clip A
            teacher_features: (B, N, D) teacher output from clip B
        """
        from scipy.optimize import linear_sum_assignment

        B, N, D = student_features.shape
        s_norm = F.normalize(student_features, dim=-1)
        t_norm = F.normalize(teacher_features, dim=-1)

        total_loss = torch.tensor(0.0, device=student_features.device)

        for b in range(B):
            # Cosine similarity matrix
            sim = s_norm[b] @ t_norm[b].T  # (N, N)

            # Hungarian matching (minimize negative similarity = maximize similarity)
            cost = -sim.detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost)

            # Loss = negative mean similarity of matched pairs
            matched_sim = sim[row_idx, col_idx]
            total_loss = total_loss - matched_sim.mean()

        return total_loss / B


class SFDRLoss(nn.Module):
    """Stochastic Feature Diversity Regularizer.

    Prevents representation collapse by encouraging diverse feature
    representations across patches. Combines:
    1. Negative variance of pairwise similarities (maximize variance)
    2. Covariance regularization (minimize off-diagonal covariance)
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) token embeddings
        """
        B, N, D = features.shape
        flat = features.reshape(B * N, D)

        # 1. Variance of pairwise similarities
        # Sample a subset for efficiency if too many tokens
        if flat.shape[0] > 512:
            idx = torch.randperm(flat.shape[0], device=flat.device)[:512]
            sample = flat[idx]
        else:
            sample = flat

        sample_norm = F.normalize(sample, dim=-1)
        sims = sample_norm @ sample_norm.T
        # Exclude diagonal
        mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
        var_loss = -sims[mask].var()

        # 2. Covariance regularization (off-diagonal)
        centered = flat - flat.mean(dim=0)
        cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        cov_loss = off_diag.pow(2).sum() / D

        return var_loss + cov_loss


class VJEPACombinedLoss(nn.Module):
    """Combines all V-JEPA losses with configurable weights.

    Returns (total_loss, loss_dict) for logging.
    """

    def __init__(
        self,
        lambda_jepa: float = 1.0,
        lambda_affinity: float = 0.5,
        lambda_cross: float = 0.3,
        lambda_sfdr: float = 0.1,
        affinity_teacher_temp: float = 0.04,
        affinity_student_temp: float = 0.1,
    ):
        super().__init__()
        self.lambda_jepa = lambda_jepa
        self.lambda_affinity = lambda_affinity
        self.lambda_cross = lambda_cross
        self.lambda_sfdr = lambda_sfdr

        self.jepa_loss = JEPALoss()
        self.affinity_loss = AffinityLoss(affinity_teacher_temp, affinity_student_temp)
        self.cross_loss = CrossMatchingLoss()
        self.sfdr_loss = SFDRLoss()

    def forward(
        self,
        predicted: torch.Tensor,
        teacher_target: torch.Tensor,
        masked_indices: torch.LongTensor,
        cross_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predicted: (B, N_total, D) predictor output
            teacher_target: (B, N_total, D) teacher output for same clip
            masked_indices: (N_masked,) masked positions
            cross_teacher: (B, N_total, D) teacher output for cross-video clip,
                or None if no cross-video pair available

        Returns:
            (total_loss, {"jepa": ..., "affinity": ..., "cross": ..., "sfdr": ...})
        """
        loss_dict: dict[str, float] = {}

        # L_jepa: masked prediction
        l_jepa = self.jepa_loss(predicted, teacher_target, masked_indices)
        total = self.lambda_jepa * l_jepa
        loss_dict["jepa"] = l_jepa.item()

        # L_affinity: relational distillation
        if self.lambda_affinity > 0:
            l_aff = self.affinity_loss(predicted, teacher_target)
            total = total + self.lambda_affinity * l_aff
            loss_dict["affinity"] = l_aff.item()

        # L_cross: cross-video Hungarian matching
        if self.lambda_cross > 0 and cross_teacher is not None:
            l_cross = self.cross_loss(predicted, cross_teacher)
            total = total + self.lambda_cross * l_cross
            loss_dict["cross"] = l_cross.item()

        # L_sfdr: feature diversity
        if self.lambda_sfdr > 0:
            l_sfdr = self.sfdr_loss(predicted)
            total = total + self.lambda_sfdr * l_sfdr
            loss_dict["sfdr"] = l_sfdr.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
