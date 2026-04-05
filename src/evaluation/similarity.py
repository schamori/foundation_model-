"""
Patch similarity evaluation.

Computes structure-weighted spread (SWS) and augmentation stability scores
on sample images using the pretrained backbone.

Absorbed from own experiments/similarity_visualization/visualize_similarity.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import BaseEvaluator, register_evaluator

if True:  # TYPE_CHECKING guard that still runs
    from ..config import Config


# ---------------------------------------------------------------------------
# Core similarity computation
# ---------------------------------------------------------------------------

class PatchSimilarityVisualizer:
    """Extract dense features and compute patch-level cosine similarity maps."""

    def __init__(self, backbone: torch.nn.Module, device: str = "cuda"):
        self.device = device
        self.model = backbone.to(device).eval()
        self.features = None
        self._register_hook()

    def _register_hook(self):
        # HuggingFace ConvNext: encoder.stages[-1]
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "stages"):
            self.model.encoder.stages[-1].register_forward_hook(
                lambda mod, inp, out: setattr(self, "features", out))
        # timm models: stages[-1] or feature_info
        elif hasattr(self.model, "stages"):
            self.model.stages[-1].register_forward_hook(
                lambda mod, inp, out: setattr(self, "features", out))
        else:
            print(f"[similarity] Warning: could not find stages for hook, features may be None")

    def extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _ = self.model(img_tensor.to(self.device))
        return self.features

    def compute_similarity_map(self, features: torch.Tensor, qh: int, qw: int) -> np.ndarray:
        B, C, H, W = features.shape
        query = features[0, :, qh, qw]
        flat = features[0].permute(1, 2, 0).reshape(-1, C)
        qn = F.normalize(query.unsqueeze(0), dim=1)
        fn = F.normalize(flat, dim=1)
        sim = torch.mm(fn, qn.T).squeeze().reshape(H, W)
        return sim.cpu().numpy()


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _sobel_grad_magnitude(img_rgb: np.ndarray) -> np.ndarray:
    gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gray_t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(gray_t, sx, padding=1)
    gy = F.conv2d(gray_t, sy, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2).squeeze().numpy()


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _grid_points(n: int, rows: int = 4, cols: int = 5, margin: float = 0.1) -> list[tuple[float, float]]:
    if rows * cols != n:
        rows = int(round(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
    ys = np.linspace(margin, 1.0 - margin, rows)
    xs = np.linspace(margin, 1.0 - margin, cols)
    return [(float(y), float(x)) for y in ys for x in xs][:n]


def _structure_weighted_spread(sim_map: np.ndarray, img_rgb: np.ndarray, top_k: float = 0.20) -> float:
    h, w = sim_map.shape
    edge = _sobel_grad_magnitude(img_rgb)
    edge = edge / (edge.max() + 1e-8)
    threshold = np.percentile(sim_map, (1 - top_k) * 100)
    mask = sim_map >= threshold
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    weights = edge[ys, xs] + 0.05
    yn, xn = ys / h, xs / w
    ym = np.average(yn, weights=weights)
    xm = np.average(xn, weights=weights)
    return float(np.sqrt(np.average((yn - ym) ** 2, weights=weights)) +
                 np.sqrt(np.average((xn - xm) ** 2, weights=weights)))


def compute_similarity_scores(
    backbone: torch.nn.Module,
    image_path: str | Path,
    device: str = "cuda",
    n_points: int = 20,
    output_size: int = 512,
) -> dict:
    """Compute SWS and augmentation stability for one image."""
    viz = PatchSimilarityVisualizer(backbone, device)
    img = Image.open(image_path).convert("RGB")
    base_transform = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = base_transform(img).unsqueeze(0)
    img_display = np.array(img.resize((output_size, output_size)))
    features = viz.extract_features(img_tensor)
    if features is None:
        return {"structure_weighted_spread": 0.0, "augmentation_stability": {}}
    _, _, fh, fw = features.shape
    qps = _grid_points(n_points)

    # SWS
    sws_scores = []
    for qy, qx in qps:
        fy, fx = int(np.clip(qy * fh, 0, fh - 1)), int(np.clip(qx * fw, 0, fw - 1))
        sm = viz.compute_similarity_map(features, fy, fx)
        sm_up = F.interpolate(torch.tensor(sm).unsqueeze(0).unsqueeze(0),
                              size=(output_size, output_size), mode="bilinear",
                              align_corners=False).squeeze().numpy()
        sws_scores.append(_structure_weighted_spread(sm_up, img_display))

    # Augmentation stability
    aug_transforms = {
        "hflip": transforms.Compose([transforms.Resize((output_size, output_size)),
                                     transforms.RandomHorizontalFlip(p=1.0),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "color_jitter": transforms.Compose([transforms.Resize((output_size, output_size)),
                                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "blur": transforms.Compose([transforms.Resize((output_size, output_size)),
                                    transforms.GaussianBlur(5, (0.1, 1.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    stability = {}
    for name, aug_t in aug_transforms.items():
        aug_tensor = aug_t(img).unsqueeze(0)
        feat_aug = viz.extract_features(aug_tensor)
        if feat_aug is None:
            stability[name] = 0.0
            continue
        _, _, fh_a, fw_a = feat_aug.shape
        per_q = []
        for qy, qx in qps:
            qy_a, qx_a = qy, qx
            if name == "hflip":
                qx_a = 1.0 - qx
            fy = int(np.clip(qy * fh, 0, fh - 1))
            fx = int(np.clip(qx * fw, 0, fw - 1))
            sm = viz.compute_similarity_map(features, fy, fx)
            sm_up = F.interpolate(torch.tensor(sm).unsqueeze(0).unsqueeze(0),
                                  size=(output_size, output_size), mode="bilinear",
                                  align_corners=False).squeeze().numpy()
            fy_a = int(np.clip(qy_a * fh_a, 0, fh_a - 1))
            fx_a = int(np.clip(qx_a * fw_a, 0, fw_a - 1))
            sm_a = viz.compute_similarity_map(feat_aug, fy_a, fx_a)
            sm_a_up = F.interpolate(torch.tensor(sm_a).unsqueeze(0).unsqueeze(0),
                                    size=(output_size, output_size), mode="bilinear",
                                    align_corners=False).squeeze().numpy()
            per_q.append(_pearson_corr(sm_up, sm_a_up))
        stability[name] = float(np.mean(per_q)) if per_q else 0.0

    return {
        "structure_weighted_spread": float(np.mean(sws_scores)) if sws_scores else 0.0,
        "augmentation_stability": stability,
    }


# ---------------------------------------------------------------------------
# Evaluator registration
# ---------------------------------------------------------------------------

@register_evaluator
class SimilarityEvaluator(BaseEvaluator):
    """Evaluate patch similarity quality on train and validation images."""

    name = "similarity"
    N_SAMPLES = 50  # images to sample per split

    def _discover_images(self, root: Path) -> list[Path]:
        """Collect all image paths under root (flat or nested video dirs)."""
        from ..data.dataset import discover_frames
        videos = discover_frames(root, self.cfg.exclude_folders)
        return [p for frames in videos.values() for p in frames]

    def _eval_split(self, backbone, images: list[Path], split: str) -> dict:
        """Score a random sample from one split."""
        import random as _rng
        sampled = _rng.sample(images, min(self.N_SAMPLES, len(images)))
        all_scores = []
        for img_path in sampled:
            scores = compute_similarity_scores(backbone, img_path, device=self.cfg.device)
            all_scores.append(scores)
        if not all_scores:
            return {}
        avg_sws = float(np.mean([s["structure_weighted_spread"] for s in all_scores]))
        avg_stability = {}
        for key in all_scores[0].get("augmentation_stability", {}):
            vals = [s["augmentation_stability"].get(key, 0) for s in all_scores]
            avg_stability[key] = float(np.mean(vals))
        return {f"{split}_sws": avg_sws, f"{split}_augmentation_stability": avg_stability}

    def evaluate_checkpoint(self, checkpoint_path: Path, epoch: int) -> dict:
        from ..model.backbone import load_backbone_from_checkpoint
        backbone = load_backbone_from_checkpoint(self.cfg, checkpoint_path)

        result: dict = {"epoch": epoch}
        n_images = 0

        # Train split
        train_root = self.cfg.frames_root
        if train_root and train_root.is_dir():
            train_imgs = self._discover_images(train_root)
            n_images += len(train_imgs)
            if train_imgs:
                result.update(self._eval_split(backbone, train_imgs, "train"))

        # Validation split
        val_root = getattr(self.cfg, "eval_frames_root", None)
        if val_root and val_root.is_dir():
            val_imgs = self._discover_images(val_root)
            n_images += len(val_imgs)
            if val_imgs:
                result.update(self._eval_split(backbone, val_imgs, "val"))

        if n_images == 0:
            return {"error": "no images found in frames_root or eval_frames_root"}

        self.results.append(result)
        return result
