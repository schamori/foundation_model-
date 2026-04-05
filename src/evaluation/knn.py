"""
k-NN evaluation on labeled frames.

Extracts embeddings from the teacher backbone, runs k-NN classification
using frame-level phase/instrument labels, and reports top-1 accuracy.

Labels are discovered automatically from the directory structure:
    frames_root/<label_name>/<video_name>/frame_*.jpg
or from an optional labels_json mapping frame_path → label.

Add to training eval by ensuring this module is imported in __init__.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import BaseEvaluator, register_evaluator

if TYPE_CHECKING:
    from ..config import Config


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def extract_embeddings(
    backbone: torch.nn.Module,
    image_paths: list[Path],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract L2-normalised embeddings for a list of image paths."""
    backbone = backbone.to(device).eval()
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(IMAGENET_TRANSFORM(img))
            batch = torch.stack(tensors).to(device)

            out = backbone(batch)
            # Pool to (B, D)
            if hasattr(out, "pooler_output"):
                feats = out.pooler_output.squeeze(-1).squeeze(-1)
            elif isinstance(out, torch.Tensor):
                if out.dim() == 4:
                    feats = out.mean(dim=[2, 3])
                else:
                    feats = out
            else:
                feats = out.last_hidden_state[:, 0]  # ViT CLS token

            all_feats.append(F.normalize(feats.float(), dim=1).cpu())

    return torch.cat(all_feats, dim=0)  # (N, D)


def knn_accuracy(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """
    Weighted k-NN classifier (as in DINO eval).
    Uses cosine similarity + temperature softmax weighting.
    """
    n_classes = int(train_labels.max().item()) + 1
    correct = 0

    # Process in chunks to avoid OOM
    chunk = 512
    for start in range(0, len(val_feats), chunk):
        q = val_feats[start : start + chunk]  # (B, D)
        sim = torch.mm(q, train_feats.T)       # (B, N_train)
        sim_k, idx_k = sim.topk(k, dim=1)     # (B, k)

        # Weighted vote
        weights = (sim_k / temperature).exp()  # (B, k)
        votes = torch.zeros(len(q), n_classes)
        for j in range(k):
            labels_j = train_labels[idx_k[:, j]]  # (B,)
            for b in range(len(q)):
                votes[b, labels_j[b]] += weights[b, j]

        pred = votes.argmax(dim=1)
        correct += (pred == val_labels[start : start + chunk]).sum().item()

    return correct / len(val_feats)


def _discover_labeled_frames(
    labels_root: Path,
    max_per_class: int = 500,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[int], list[Path], list[int]] | None:
    """
    Discover frames from a directory structured as:
        labels_root/<class_name>/<video_or_flat>/<frame>.jpg

    Returns (train_paths, train_labels, val_paths, val_labels) or None.
    """
    rng = np.random.default_rng(seed)
    class_dirs = sorted([d for d in labels_root.iterdir() if d.is_dir()])
    if not class_dirs:
        return None

    all_paths: list[Path] = []
    all_labels: list[int] = []

    for label_idx, class_dir in enumerate(class_dirs):
        frames = sorted(class_dir.rglob("*.jpg")) + sorted(class_dir.rglob("*.png"))
        if not frames:
            continue
        if len(frames) > max_per_class:
            frames = [frames[i] for i in rng.choice(len(frames), max_per_class, replace=False)]
        all_paths.extend(frames)
        all_labels.extend([label_idx] * len(frames))

    if not all_paths:
        return None

    indices = np.arange(len(all_paths))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        [all_paths[i] for i in train_idx],
        [all_labels[i] for i in train_idx],
        [all_paths[i] for i in val_idx],
        [all_labels[i] for i in val_idx],
    )


@register_evaluator
class KNNEvaluator(BaseEvaluator):
    """
    k-NN accuracy on labeled frames.

    Configure via cfg.knn_labels_root (Path to class-structured directory).
    If not set, this evaluator silently skips.

    To add more label sets, subclass and override `labels_root`.
    """

    name = "knn"

    @property
    def labels_root(self) -> Path | None:
        return getattr(self.cfg, "knn_labels_root", None)

    def evaluate_checkpoint(self, checkpoint_path: Path, epoch: int) -> dict:
        if self.labels_root is None or not Path(self.labels_root).exists():
            return {"skipped": "knn_labels_root not set or not found"}

        from ..model.backbone import load_backbone_from_checkpoint

        result = _discover_labeled_frames(Path(self.labels_root))
        if result is None:
            return {"error": "no labeled frames found"}

        train_paths, train_labels_list, val_paths, val_labels_list = result
        backbone = load_backbone_from_checkpoint(self.cfg, checkpoint_path)

        train_feats = extract_embeddings(backbone, train_paths, self.cfg.device)
        val_feats = extract_embeddings(backbone, val_paths, self.cfg.device)

        train_labels = torch.tensor(train_labels_list)
        val_labels = torch.tensor(val_labels_list)

        metrics = {}
        for k in [1, 5, 20]:
            if k <= len(train_paths):
                acc = knn_accuracy(train_feats, train_labels, val_feats, val_labels, k=k)
                metrics[f"top1_k{k}"] = round(acc, 4)

        metrics["epoch"] = epoch
        metrics["n_train"] = len(train_paths)
        metrics["n_val"] = len(val_paths)
        self.results.append(metrics)
        return metrics
