"""
Training dataset for surgical frame pretraining.

Supports both DINO (single-frame with augmentation pool) and V-JEPA (video clip)
modes. Handles temporal activity-weighted sampling, cross-video pair lookup, and
temporal neighbor retrieval.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from ..utils.augmentations import PoolAwareMultiCropAugmentation

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Frame discovery (mirrors src/feature_extractor/extract.py:_discover_frames)
# ---------------------------------------------------------------------------

def discover_frames(
    frames_root: Path,
    exclude_folders: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover frames grouped by video key.

    Handles both flat (frames_root/video/*.jpg) and nested
    (frames_root/category/video/*.jpg) layouts.

    Returns {video_key: [sorted frame paths]}.
    """
    exclude = set(exclude_folders or [])
    videos: dict[str, list[Path]] = {}

    for item in sorted(frames_root.iterdir()):
        if not item.is_dir() or item.name in exclude:
            continue
        has_images = any(f.suffix.lower() in IMAGE_SUFFIXES for f in item.iterdir() if f.is_file())
        if has_images:
            # Flat layout: frames_root/video/*.jpg
            paths = sorted(f for f in item.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            if paths:
                videos[item.name] = paths
        else:
            # Nested layout: frames_root/category/video/*.jpg
            for vdir in sorted(item.iterdir()):
                if vdir.is_dir() and vdir.name not in exclude:
                    paths = sorted(f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
                    if paths:
                        videos[f"{item.name}/{vdir.name}"] = paths

    return videos


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurgicalFrameDataset(Dataset):
    """Training dataset for surgical SSL pretraining.

    Args:
        frames_root: directory with [category/]video_name/frame_NNNNNN.jpg
        exclude_folders: folder names to skip during discovery
        temporal_scores_path: JSON from preprocessing.py (activity scores)
        pair_index_path: JSON from preprocessing.py (cross-video pairs)
        temporal_neighbor_range: ±N adjacent frames for DINO augmentation pool
        use_cross_video_pairs: whether to include cross-video frames in pool
        activity_alpha: exponent for activity-weighted sampling (0 = uniform)
        transform: augmentation callable
        mode: "dino" (single frame + pool) or "vjepa" (video clip)
        clip_length: V-JEPA only — number of frames per clip
        clip_stride: V-JEPA only — temporal stride between frames
    """

    def __init__(
        self,
        frames_root: Path,
        exclude_folders: list[str] | None = None,
        temporal_scores_path: Path | None = None,
        pair_index_path: Path | None = None,
        temporal_neighbor_range: int = 2,
        use_cross_video_pairs: bool = True,
        activity_alpha: float = 1.0,
        transform=None,
        mode: str = "dino",
        clip_length: int = 16,
        clip_stride: int = 6,
    ):
        self.mode = mode
        self.transform = transform
        self.temporal_neighbor_range = temporal_neighbor_range
        self.use_cross_video_pairs = use_cross_video_pairs
        self.clip_length = clip_length
        self.clip_stride = clip_stride

        # Discover all frames
        video_dict = discover_frames(frames_root, exclude_folders)
        self._frames: list[Path] = []
        self._frame_video: list[str] = []  # video key for each frame
        self._video_frames: dict[str, list[int]] = defaultdict(list)

        for vkey in sorted(video_dict):
            for p in video_dict[vkey]:
                idx = len(self._frames)
                self._frames.append(p)
                self._frame_video.append(vkey)
                self._video_frames[vkey].append(idx)

        n_videos = len(self._video_frames)
        n_frames = len(self._frames)
        print(f"[dataset] {n_videos} videos, {n_frames:,} frames, mode={mode}")

        # Build path → index for pair lookup
        self._path_to_idx: dict[str, int] = {str(p): i for i, p in enumerate(self._frames)}

        # Load optional data
        self._temporal_scores = self._load_temporal_scores(temporal_scores_path)
        self._pair_index = self._load_pair_index(pair_index_path)
        # For V-JEPA: build clip index (list of valid clip start indices)
        if mode == "vjepa":
            self._clips = self._build_clip_index()
            print(f"[dataset] {len(self._clips):,} valid clips "
                  f"(length={clip_length}, stride={clip_stride})")

        self._sampling_weights = self._build_sampling_weights(activity_alpha)

    def __len__(self) -> int:
        if self.mode == "vjepa":
            return len(self._clips)
        return len(self._frames)

    def __getitem__(self, idx: int):
        if self.mode == "vjepa":
            return self._getitem_vjepa(idx)
        return self._getitem_dino(idx)

    # -- DINO mode ----------------------------------------------------------

    def _getitem_dino(self, idx: int):
        """Return augmentation pool for DINO training."""
        pool = self._build_augmentation_pool(idx)
        images = []
        for p in pool:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        if self.transform is not None:
            # PoolAwareMultiCropAugmentation expects list, MultiCropAugmentation expects single image
            if isinstance(self.transform, PoolAwareMultiCropAugmentation):
                return self.transform(images)
            return self.transform(images[0])
        return images

    def _build_augmentation_pool(self, anchor_idx: int) -> list[Path]:
        """Build pool: anchor + temporal neighbors + cross-video pairs."""
        pool = [self._frames[anchor_idx]]
        vkey = self._frame_video[anchor_idx]
        v_indices = self._video_frames[vkey]

        # Temporal neighbors
        if self.temporal_neighbor_range > 0:
            local_pos = v_indices.index(anchor_idx)
            for offset in range(-self.temporal_neighbor_range,
                                self.temporal_neighbor_range + 1):
                if offset == 0:
                    continue
                ni = local_pos + offset
                if 0 <= ni < len(v_indices):
                    pool.append(self._frames[v_indices[ni]])

        # Cross-video pairs
        if self.use_cross_video_pairs and self._pair_index is not None:
            anchor_path = str(self._frames[anchor_idx])
            matches = self._pair_index.get(anchor_path, [])
            for mp in matches[:2]:  # up to 2 cross-video frames
                if mp in self._path_to_idx:
                    pool.append(self._frames[self._path_to_idx[mp]])

        return pool

    # -- V-JEPA mode --------------------------------------------------------

    def _getitem_vjepa(self, idx: int):
        """Return a video clip as list of PIL images."""
        clip_start_idx = self._clips[idx]
        vkey = self._frame_video[clip_start_idx]
        v_indices = self._video_frames[vkey]
        local_start = v_indices.index(clip_start_idx)

        clip_images = []
        for i in range(self.clip_length):
            fi = local_start + i * self.clip_stride
            if fi < len(v_indices):
                with Image.open(self._frames[v_indices[fi]]) as img:
                    clip_images.append(img.convert("RGB"))
            else:
                # Repeat last frame if clip extends beyond video
                clip_images.append(clip_images[-1].copy())

        if self.transform is not None:
            return self.transform(clip_images)
        return clip_images

    def _build_clip_index(self) -> list[int]:
        """Find all valid clip start positions across all videos."""
        clips = []
        needed = (self.clip_length - 1) * self.clip_stride + 1
        for vkey, v_indices in self._video_frames.items():
            if len(v_indices) < needed:
                continue
            # Clip starts at every stride-th position
            for local_i in range(0, len(v_indices) - needed + 1, self.clip_stride):
                clips.append(v_indices[local_i])
        return clips

    # -- Data loading helpers -----------------------------------------------

    def _load_temporal_scores(self, path: Path | None) -> dict[str, list[float]] | None:
        if path is None or not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        # Group scores by video
        scores: dict[str, list[float]] = defaultdict(list)
        for entry in data.get("changes", []):
            scores[entry["video"]].append(entry["score"])
        print(f"[dataset] Loaded temporal scores: {len(scores)} videos")
        return dict(scores)

    def _load_pair_index(self, path: Path | None) -> dict[str, list[str]] | None:
        if path is None or not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        pairs = data.get("pairs", {})
        print(f"[dataset] Loaded pair index: {len(pairs)} frames with matches")
        return pairs

    def _build_sampling_weights(self, alpha: float) -> np.ndarray:
        """Build activity-weighted sampling distribution.

        Uses temporal scores if available, otherwise uniform.
        Weights = score(t)^alpha, normalized.
        """
        n = len(self) if self.mode == "dino" else len(self._clips)
        if alpha == 0 or self._temporal_scores is None:
            return np.ones(n, dtype=np.float64) / n

        # Map per-frame scores: use max change score for each video
        video_max_score: dict[str, float] = {}
        for vkey, scores in self._temporal_scores.items():
            video_max_score[vkey] = max(scores) if scores else 0.0

        weights = np.ones(n, dtype=np.float64)
        if self.mode == "vjepa":
            for i, clip_start in enumerate(self._clips):
                vkey = self._frame_video[clip_start]
                weights[i] = max(video_max_score.get(vkey, 0.0), 1e-8)
        else:
            for i in range(n):
                vkey = self._frame_video[i]
                weights[i] = max(video_max_score.get(vkey, 0.0), 1e-8)

        weights = weights ** alpha
        weights /= weights.sum()
        return weights

    def sampler(self) -> WeightedRandomSampler:
        """Return a weighted random sampler for DataLoader."""
        return WeightedRandomSampler(
            weights=self._sampling_weights,
            num_samples=len(self._sampling_weights),
            replacement=True,
        )
