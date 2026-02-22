#!/usr/bin/env python3
"""
Create and evaluate train/test splits for Cholec80 videos.

Pipeline requested:
1. Embed frames with ConvNeXt only (no ResNet).
2. Average frame embeddings per video.
3. L2-normalize per-video embeddings.
4. Cluster videos with spherical k-means (cosine similarity in full embedding space).
5. Stratify train/test split over cluster labels.
6. Compare label-balance metric against random and length-stratified baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ConvNextImageProcessor, ConvNextModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BASE_DIR_DEFAULT = Path("/media/HDD1/moritz/AI for Surg/")
FRAMES_DIR_DEFAULT = BASE_DIR_DEFAULT / "frames_224"
INSTRUMENT_DIR_DEFAULT = Path("/home/moritz/AI for surg/instrument detection/tool_annotations")
PHASE_DIR_DEFAULT = BASE_DIR_DEFAULT / "phase_annotations_with_time"

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_PATH_DEFAULT = SCRIPT_DIR / "video_embedding_cache.pkl"
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "split_outputs"


@dataclass
class VideoRecord:
    video_id: str
    frame_dir: Path
    frame_paths: list[Path]
    frame_count: int
    instrument_labels: list[str]
    instrument_positive_counts: np.ndarray
    instrument_total: int
    phase_count_map: dict[str, int]
    phase_total: int


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_video_sort_key(video_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", video_id)
    if match:
        return int(match.group(1)), video_id
    return 10**9, video_id


def parse_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            return [], []

        headers = [h.strip().lstrip("\ufeff") for h in first_line.rstrip("\n").split("\t")]
        rows: list[list[str]] = []

        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            values = [v.strip() for v in line.split("\t")]
            if len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            rows.append(values[: len(headers)])

    return headers, rows


def parse_instrument_annotations(path: Path) -> tuple[list[str], np.ndarray, int]:
    headers, rows = parse_tsv(path)
    if not headers:
        raise ValueError(f"Empty instrument annotation file: {path}")

    header_map = {h.lower(): i for i, h in enumerate(headers)}
    frame_idx = header_map.get("frame")
    label_indices = [i for i in range(len(headers)) if i != frame_idx]
    label_names = [headers[i] for i in label_indices]

    positive_counts = np.zeros(len(label_names), dtype=np.float64)
    for row in rows:
        for j, idx in enumerate(label_indices):
            value = row[idx] if idx < len(row) else "0"
            try:
                positive_counts[j] += float(value)
            except ValueError:
                continue

    return label_names, positive_counts, len(rows)


def parse_phase_annotations(path: Path) -> tuple[dict[str, int], int]:
    headers, rows = parse_tsv(path)
    if not headers:
        raise ValueError(f"Empty phase annotation file: {path}")

    phase_idx = None
    for i, h in enumerate(headers):
        if h.lower() == "phase":
            phase_idx = i
            break

    if phase_idx is None:
        raise ValueError(f"No 'Phase' column found in {path}")

    counts: Counter[str] = Counter()
    for row in rows:
        if phase_idx >= len(row):
            continue
        phase = row[phase_idx].strip()
        if phase:
            counts[phase] += 1

    total = int(sum(counts.values()))
    return dict(counts), total


def list_frame_paths(frame_dir: Path) -> list[Path]:
    paths = [
        p
        for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def collect_video_records(
    frames_dir: Path,
    instrument_dir: Path,
    phase_dir: Path,
    max_videos: int | None = None,
) -> tuple[list[VideoRecord], list[str], list[str]]:
    video_dirs = [
        p
        for p in frames_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"video\d+", p.name)
    ]
    video_dirs.sort(key=lambda p: numeric_video_sort_key(p.name))

    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]

    records: list[VideoRecord] = []
    reference_instrument_labels: list[str] | None = None
    all_phase_labels: set[str] = set()

    for video_dir in tqdm(video_dirs, desc="Indexing videos"):
        video_id = video_dir.name
        instrument_file = instrument_dir / f"{video_id}-tool.txt"
        phase_file = phase_dir / f"{video_id}-phase-time.txt"

        if not instrument_file.exists():
            print(f"[warn] Missing instrument annotation for {video_id}: {instrument_file}")
            continue
        if not phase_file.exists():
            print(f"[warn] Missing phase annotation for {video_id}: {phase_file}")
            continue

        frame_paths = list_frame_paths(video_dir)
        if not frame_paths:
            print(f"[warn] No frames found for {video_id}")
            continue

        instrument_labels, instrument_counts, instrument_total = parse_instrument_annotations(instrument_file)
        if reference_instrument_labels is None:
            reference_instrument_labels = instrument_labels
        elif instrument_labels != reference_instrument_labels:
            raise ValueError(
                f"Instrument labels mismatch in {instrument_file}.\n"
                f"Expected: {reference_instrument_labels}\n"
                f"Found:    {instrument_labels}"
            )

        phase_count_map, phase_total = parse_phase_annotations(phase_file)
        all_phase_labels.update(phase_count_map.keys())

        records.append(
            VideoRecord(
                video_id=video_id,
                frame_dir=video_dir,
                frame_paths=frame_paths,
                frame_count=len(frame_paths),
                instrument_labels=instrument_labels,
                instrument_positive_counts=instrument_counts,
                instrument_total=instrument_total,
                phase_count_map=phase_count_map,
                phase_total=phase_total,
            )
        )

    if not records:
        raise RuntimeError("No valid videos found after indexing.")

    assert reference_instrument_labels is not None
    phase_labels = sorted(all_phase_labels)
    return records, reference_instrument_labels, phase_labels


def sample_frame_paths(
    frame_paths: Sequence[Path],
    frame_stride: int,
    max_frames_per_video: int | None,
) -> list[Path]:
    stride = max(1, int(frame_stride))
    sampled = list(frame_paths[::stride])

    if max_frames_per_video is not None and len(sampled) > max_frames_per_video:
        idx = np.linspace(0, len(sampled) - 1, num=max_frames_per_video, dtype=np.int64)
        sampled = [sampled[i] for i in idx.tolist()]

    return sampled


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    cos = float(np.clip(cos, -1.0, 1.0))
    return 1.0 - cos


def load_convnext_embedder(
    model_name: str,
    device: torch.device,
    checkpoint_path: Path | None,
) -> tuple[Callable[[list[Image.Image]], np.ndarray], int]:
    model = ConvNextModel.from_pretrained(model_name)

    if checkpoint_path is not None:
        if checkpoint_path.exists():
            print(f"Loading ConvNeXt checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            if isinstance(state_dict, dict):
                # Strip an optional common prefix.
                if all(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[warn] Missing keys while loading checkpoint ({len(missing)} keys)")
                if unexpected:
                    print(f"[warn] Unexpected keys while loading checkpoint ({len(unexpected)} keys)")
            else:
                raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
        else:
            raise FileNotFoundError(f"ConvNeXt checkpoint not found: {checkpoint_path}")

    model = model.to(device).eval()

    processor = ConvNextImageProcessor.from_pretrained(
        model_name,
        size={"shortest_edge": 224},
        crop_size={"height": 224, "width": 224},
    )

    @torch.no_grad()
    def embed_batch(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        features = outputs.pooler_output
        if features.ndim == 4:
            features = features.squeeze(-1).squeeze(-1)
        return features.detach().cpu().numpy().astype(np.float32)

    emb_dim = int(model.config.hidden_sizes[-1])
    return embed_batch, emb_dim


def batch_mean_embedding(
    frame_paths: Sequence[Path],
    embed_batch_fn: Callable[[list[Image.Image]], np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, int]:
    running_sum: np.ndarray | None = None
    total_count = 0

    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images: list[Image.Image] = []

        for frame_path in batch_paths:
            try:
                images.append(load_image_rgb(frame_path))
            except Exception as exc:
                print(f"[warn] Failed to load frame {frame_path}: {exc}")

        if not images:
            continue

        features = embed_batch_fn(images)
        if features.ndim != 2:
            raise ValueError(f"Expected feature batch shape [B, D], got {features.shape}")

        batch_sum = features.astype(np.float64).sum(axis=0)
        if running_sum is None:
            running_sum = batch_sum
        else:
            running_sum += batch_sum

        total_count += features.shape[0]

    if running_sum is None or total_count == 0:
        raise RuntimeError("Could not extract any features for this video.")

    mean_embedding = (running_sum / float(total_count)).astype(np.float32)
    return mean_embedding, total_count


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception as exc:
        print(f"[warn] Failed to read cache {cache_path}: {exc}")

    return {}


def save_cache(cache_path: Path, cache_obj: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache_obj, f)


def build_cache_key(name: str, settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


def compute_video_embeddings(
    records: Sequence[VideoRecord],
    embed_batch_fn: Callable[[list[Image.Image]], np.ndarray],
    embedding_dim: int,
    frame_stride: int,
    max_frames_per_video: int | None,
    batch_size: int,
    cache: dict,
    cache_key: str,
    force_recompute: bool,
    desc: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    per_video: dict[str, np.ndarray] = {}
    used_frames: dict[str, int] = {}

    cache_block = cache.get(cache_key, {})
    if not isinstance(cache_block, dict):
        cache_block = {}

    for record in tqdm(records, desc=desc):
        cached = cache_block.get(record.video_id)
        if (not force_recompute) and cached is not None:
            arr = np.asarray(cached, dtype=np.float32)
            if arr.shape == (embedding_dim,):
                per_video[record.video_id] = arr
                sampled = sample_frame_paths(record.frame_paths, frame_stride, max_frames_per_video)
                used_frames[record.video_id] = len(sampled)
                continue

        sampled = sample_frame_paths(record.frame_paths, frame_stride, max_frames_per_video)
        if not sampled:
            raise RuntimeError(f"No sampled frames for {record.video_id}")

        mean_emb, n_used = batch_mean_embedding(
            frame_paths=sampled,
            embed_batch_fn=embed_batch_fn,
            batch_size=batch_size,
        )

        per_video[record.video_id] = mean_emb
        used_frames[record.video_id] = n_used
        cache_block[record.video_id] = mean_emb

    cache[cache_key] = cache_block
    return per_video, used_frames


def spherical_kmeans(
    x: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {x.shape}")
    n_samples = x.shape[0]

    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})")

    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_obj = float("inf")

    for init_id in range(n_init):
        rng = np.random.default_rng(seed + init_id)
        centroid_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = x[centroid_idx].copy()
        centroids = l2_normalize_rows(centroids)

        last_labels = None

        for _ in range(max_iter):
            sims = x @ centroids.T  # cosine since both sides are normalized
            labels = np.argmax(sims, axis=1)

            if last_labels is not None and np.array_equal(labels, last_labels):
                break

            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(n_clusters):
                members = x[labels == cluster_id]
                if members.shape[0] == 0:
                    new_centroids[cluster_id] = x[rng.integers(0, n_samples)]
                else:
                    new_centroids[cluster_id] = members.mean(axis=0)

            new_centroids = l2_normalize_rows(new_centroids)
            shift = float(np.linalg.norm(centroids - new_centroids))
            centroids = new_centroids
            last_labels = labels

            if shift < tol:
                break

        sims = x @ centroids.T
        labels = np.argmax(sims, axis=1)
        max_sim = sims[np.arange(n_samples), labels]
        objective = float(np.mean(1.0 - max_sim))

        if objective < best_obj:
            best_obj = objective
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    if best_labels is None or best_centroids is None:
        raise RuntimeError("spherical_kmeans failed to produce a solution")

    return best_labels.astype(np.int64), best_centroids, best_obj


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or values.size == 0:
        return np.zeros_like(values, dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, num=n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if edges.size <= 2:
        return np.zeros_like(values, dtype=np.int64)

    bins = np.digitize(values, edges[1:-1], right=True)
    return bins.astype(np.int64)


def stratified_split_indices(
    strata: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_videos = int(strata.shape[0])
    target_test = int(round(n_videos * test_size))

    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, group in enumerate(strata.tolist()):
        by_group[int(group)].append(idx)

    for group_indices in by_group.values():
        rng.shuffle(group_indices)

    groups = sorted(by_group.keys())
    exact = {g: len(by_group[g]) * test_size for g in groups}
    alloc = {g: min(len(by_group[g]), int(math.floor(exact[g]))) for g in groups}

    current = int(sum(alloc.values()))

    if current < target_test:
        remainder_order = sorted(groups, key=lambda g: exact[g] - math.floor(exact[g]), reverse=True)
        to_add = target_test - current
        while to_add > 0:
            progressed = False
            for g in remainder_order:
                if alloc[g] < len(by_group[g]):
                    alloc[g] += 1
                    to_add -= 1
                    progressed = True
                    if to_add == 0:
                        break
            if not progressed:
                break

    elif current > target_test:
        remainder_order = sorted(groups, key=lambda g: exact[g] - math.floor(exact[g]))
        to_remove = current - target_test
        while to_remove > 0:
            progressed = False
            for g in remainder_order:
                if alloc[g] > 0:
                    alloc[g] -= 1
                    to_remove -= 1
                    progressed = True
                    if to_remove == 0:
                        break
            if not progressed:
                break

    test_idx_list: list[int] = []
    for g in groups:
        test_idx_list.extend(by_group[g][: alloc[g]])

    test_idx = np.array(sorted(test_idx_list), dtype=np.int64)

    if test_idx.size != target_test:
        mask = np.ones(n_videos, dtype=bool)
        mask[test_idx] = False
        remaining = np.flatnonzero(mask)

        if test_idx.size < target_test:
            need = target_test - test_idx.size
            add = rng.choice(remaining, size=need, replace=False)
            test_idx = np.sort(np.concatenate([test_idx, add.astype(np.int64)]))
        else:
            drop = rng.choice(test_idx, size=test_idx.size - target_test, replace=False)
            keep_mask = np.ones(test_idx.size, dtype=bool)
            drop_set = set(drop.tolist())
            for i, idx in enumerate(test_idx.tolist()):
                if idx in drop_set:
                    keep_mask[i] = False
            test_idx = test_idx[keep_mask]

    train_mask = np.ones(n_videos, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def random_split_indices(n_videos: int, test_size: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_test = int(round(n_videos * test_size))
    test_idx = np.sort(rng.choice(np.arange(n_videos), size=n_test, replace=False))
    train_mask = np.ones(n_videos, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def build_phase_count_matrix(records: Sequence[VideoRecord], phase_labels: Sequence[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(phase_labels)}
    matrix = np.zeros((len(records), len(phase_labels)), dtype=np.float64)

    for i, record in enumerate(records):
        for phase_label, count in record.phase_count_map.items():
            matrix[i, label_to_idx[phase_label]] = float(count)

    return matrix


def aggregate_label_distributions(
    indices: np.ndarray,
    instrument_positive_counts: np.ndarray,
    instrument_totals: np.ndarray,
    phase_count_matrix: np.ndarray,
    phase_totals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inst_sum = instrument_positive_counts[indices].sum(axis=0)
    inst_total = float(np.maximum(instrument_totals[indices].sum(), 1.0))
    inst_dist = inst_sum / inst_total

    phase_sum = phase_count_matrix[indices].sum(axis=0)
    phase_total = float(np.maximum(phase_totals[indices].sum(), 1.0))
    phase_dist = phase_sum / phase_total

    return inst_dist, phase_dist


def evaluate_split(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    instrument_positive_counts: np.ndarray,
    instrument_totals: np.ndarray,
    phase_count_matrix: np.ndarray,
    phase_totals: np.ndarray,
) -> dict:
    inst_train, phase_train = aggregate_label_distributions(
        indices=train_idx,
        instrument_positive_counts=instrument_positive_counts,
        instrument_totals=instrument_totals,
        phase_count_matrix=phase_count_matrix,
        phase_totals=phase_totals,
    )
    inst_test, phase_test = aggregate_label_distributions(
        indices=test_idx,
        instrument_positive_counts=instrument_positive_counts,
        instrument_totals=instrument_totals,
        phase_count_matrix=phase_count_matrix,
        phase_totals=phase_totals,
    )

    inst_diff = np.abs(inst_train - inst_test)
    phase_diff = np.abs(phase_train - phase_test)

    instrument_mae = float(inst_diff.mean())
    phase_mae = float(phase_diff.mean())
    combined_metric = 0.5 * instrument_mae + 0.5 * phase_mae

    return {
        "combined_metric": combined_metric,
        "instrument_mae": instrument_mae,
        "phase_mae": phase_mae,
        "instrument_train": inst_train,
        "instrument_test": inst_test,
        "instrument_abs_diff": inst_diff,
        "phase_train": phase_train,
        "phase_test": phase_test,
        "phase_abs_diff": phase_diff,
    }


def embedding_balance_metric(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_embeddings: np.ndarray,
    lengths: np.ndarray,
) -> float:
    emb_train = normalized_embeddings[train_idx].mean(axis=0)
    emb_test = normalized_embeddings[test_idx].mean(axis=0)

    emb_gap = cosine_distance(emb_train, emb_test)

    length_std = float(np.std(lengths))
    length_std = max(length_std, 1e-6)
    length_gap = abs(float(lengths[train_idx].mean()) - float(lengths[test_idx].mean())) / length_std

    return 0.9 * emb_gap + 0.1 * length_gap


def top_diff_pairs(labels: Sequence[str], diffs: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    pairs = [(label, float(diff)) for label, diff in zip(labels, diffs.tolist())]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:k]


def ndarray_to_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in arr.tolist()]


def make_split_payload(
    strategy_name: str,
    video_ids: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    evaluation: dict,
    instrument_labels: Sequence[str],
    phase_labels: Sequence[str],
) -> dict:
    return {
        "strategy": strategy_name,
        "train_videos": [video_ids[i] for i in train_idx.tolist()],
        "test_videos": [video_ids[i] for i in test_idx.tolist()],
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "metrics": {
            "combined_metric": float(evaluation["combined_metric"]),
            "instrument_mae": float(evaluation["instrument_mae"]),
            "phase_mae": float(evaluation["phase_mae"]),
        },
        "instrument": {
            "labels": list(instrument_labels),
            "train": ndarray_to_list(evaluation["instrument_train"]),
            "test": ndarray_to_list(evaluation["instrument_test"]),
            "abs_diff": ndarray_to_list(evaluation["instrument_abs_diff"]),
            "top_abs_diff": top_diff_pairs(instrument_labels, evaluation["instrument_abs_diff"], k=5),
        },
        "phase": {
            "labels": list(phase_labels),
            "train": ndarray_to_list(evaluation["phase_train"]),
            "test": ndarray_to_list(evaluation["phase_test"]),
            "abs_diff": ndarray_to_list(evaluation["phase_abs_diff"]),
            "top_abs_diff": top_diff_pairs(phase_labels, evaluation["phase_abs_diff"], k=5),
        },
    }


def resolve_checkpoint(arg_value: str) -> Path | None:
    if arg_value.strip().lower() == "none":
        return None
    return Path(arg_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cholec80 split builder and evaluator (spherical k-means)")

    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR_DEFAULT)
    parser.add_argument("--instrument-dir", type=Path, default=INSTRUMENT_DIR_DEFAULT)
    parser.add_argument("--phase-dir", type=Path, default=PHASE_DIR_DEFAULT)

    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=48)

    parser.add_argument("--convnext-model", type=str, default="facebook/convnext-large-224-22k-1k")
    parser.add_argument(
        "--convnext-checkpoint",
        type=str,
        default="none",
        help="Path to a local ConvNeXt checkpoint, or 'none'.",
    )

    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--kmeans-n-init", type=int, default=25)
    parser.add_argument("--kmeans-max-iter", type=int, default=200)
    parser.add_argument("--kmeans-tol", type=float, default=1e-4)

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--custom-search-trials", type=int, default=512)
    parser.add_argument("--length-bins", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH_DEFAULT)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)

    parser.add_argument("--max-videos", type=int, default=None, help="Debug only: limit number of videos.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    frames_dir = Path(args.frames_dir)
    instrument_dir = Path(args.instrument_dir)
    phase_dir = Path(args.phase_dir)

    for path, label in [
        (frames_dir, "frames-dir"),
        (instrument_dir, "instrument-dir"),
        (phase_dir, "phase-dir"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Frames dir: {frames_dir}")
    print(f"Instrument annotations: {instrument_dir}")
    print(f"Phase annotations: {phase_dir}")

    records, instrument_labels, phase_labels = collect_video_records(
        frames_dir=frames_dir,
        instrument_dir=instrument_dir,
        phase_dir=phase_dir,
        max_videos=args.max_videos,
    )

    if len(records) < 2:
        raise RuntimeError("Need at least 2 videos to build a split.")
    if args.n_clusters > len(records):
        raise ValueError(f"--n-clusters ({args.n_clusters}) cannot exceed number of videos ({len(records)})")

    print(f"Indexed {len(records)} videos")
    print(f"Instrument labels ({len(instrument_labels)}): {instrument_labels}")
    print(f"Phase labels ({len(phase_labels)}): {phase_labels}")

    checkpoint_path = resolve_checkpoint(args.convnext_checkpoint)
    if checkpoint_path is None:
        print("ConvNeXt checkpoint: none (using model from --convnext-model)")
    else:
        print(f"ConvNeXt checkpoint: {checkpoint_path}")

    cache = load_cache(args.cache_path)

    print("\nLoading ConvNeXt embedder...")
    convnext_embed_batch, embedding_dim = load_convnext_embedder(
        model_name=args.convnext_model,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    ckpt_signature = "none"
    if checkpoint_path is not None and checkpoint_path.exists():
        stat = checkpoint_path.stat()
        ckpt_signature = f"{checkpoint_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"

    cache_key = build_cache_key(
        "convnext_only",
        {
            "frame_stride": int(args.frame_stride),
            "max_frames_per_video": args.max_frames_per_video,
            "model": args.convnext_model,
            "checkpoint": ckpt_signature,
            "embedding_dim": embedding_dim,
        },
    )

    print("Computing/Loading ConvNeXt video embeddings...")
    convnext_per_video, sampled_frames = compute_video_embeddings(
        records=records,
        embed_batch_fn=convnext_embed_batch,
        embedding_dim=embedding_dim,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        batch_size=args.batch_size,
        cache=cache,
        cache_key=cache_key,
        force_recompute=args.force_recompute,
        desc="ConvNeXt embeddings",
    )

    save_cache(args.cache_path, cache)
    print(f"Saved embedding cache: {args.cache_path}")

    video_ids = [record.video_id for record in records]
    lengths = np.array([record.frame_count for record in records], dtype=np.float64)

    embeddings = np.stack([convnext_per_video[v] for v in video_ids], axis=0)
    normalized_embeddings = l2_normalize_rows(embeddings)

    instrument_positive_counts = np.stack(
        [record.instrument_positive_counts for record in records], axis=0
    ).astype(np.float64)
    instrument_totals = np.array([record.instrument_total for record in records], dtype=np.float64)

    phase_count_matrix = build_phase_count_matrix(records, phase_labels)
    phase_totals = np.array([record.phase_total for record in records], dtype=np.float64)

    print("Running spherical k-means clustering...")
    cluster_labels, centroids, cluster_objective = spherical_kmeans(
        x=normalized_embeddings,
        n_clusters=args.n_clusters,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        tol=args.kmeans_tol,
        seed=args.seed,
    )
    cluster_counts = Counter(cluster_labels.tolist())
    print(f"Spherical k-means objective (mean cosine distance): {cluster_objective:.6f}")
    print(f"Cluster counts: {dict(sorted(cluster_counts.items()))}")

    best_custom: tuple[np.ndarray, np.ndarray] | None = None
    best_custom_balance = float("inf")

    for trial in range(args.custom_search_trials):
        rng = np.random.default_rng(args.seed + 10_000 + trial)
        train_idx, test_idx = stratified_split_indices(
            strata=cluster_labels,
            test_size=args.test_size,
            rng=rng,
        )
        balance = embedding_balance_metric(
            train_idx=train_idx,
            test_idx=test_idx,
            normalized_embeddings=normalized_embeddings,
            lengths=lengths,
        )

        if balance < best_custom_balance:
            best_custom_balance = balance
            best_custom = (train_idx, test_idx)

    if best_custom is None:
        raise RuntimeError("Failed to build custom split")

    custom_train_idx, custom_test_idx = best_custom
    custom_eval = evaluate_split(
        train_idx=custom_train_idx,
        test_idx=custom_test_idx,
        instrument_positive_counts=instrument_positive_counts,
        instrument_totals=instrument_totals,
        phase_count_matrix=phase_count_matrix,
        phase_totals=phase_totals,
    )

    length_strata = quantile_bins(lengths, n_bins=max(2, args.length_bins))
    rng_length = np.random.default_rng(args.seed + 20_000)
    length_train_idx, length_test_idx = stratified_split_indices(
        strata=length_strata,
        test_size=args.test_size,
        rng=rng_length,
    )
    length_eval = evaluate_split(
        train_idx=length_train_idx,
        test_idx=length_test_idx,
        instrument_positive_counts=instrument_positive_counts,
        instrument_totals=instrument_totals,
        phase_count_matrix=phase_count_matrix,
        phase_totals=phase_totals,
    )

    random_metrics = np.zeros(args.random_trials, dtype=np.float64)
    random_instrument_mae = np.zeros(args.random_trials, dtype=np.float64)
    random_phase_mae = np.zeros(args.random_trials, dtype=np.float64)

    for trial in tqdm(range(args.random_trials), desc="Random baseline"):
        rng = np.random.default_rng(args.seed + 30_000 + trial)
        train_idx, test_idx = random_split_indices(
            n_videos=len(records),
            test_size=args.test_size,
            rng=rng,
        )
        e = evaluate_split(
            train_idx=train_idx,
            test_idx=test_idx,
            instrument_positive_counts=instrument_positive_counts,
            instrument_totals=instrument_totals,
            phase_count_matrix=phase_count_matrix,
            phase_totals=phase_totals,
        )
        random_metrics[trial] = e["combined_metric"]
        random_instrument_mae[trial] = e["instrument_mae"]
        random_phase_mae[trial] = e["phase_mae"]

    random_summary = {
        "n_trials": int(args.random_trials),
        "combined_metric_mean": float(np.mean(random_metrics)),
        "combined_metric_std": float(np.std(random_metrics)),
        "combined_metric_median": float(np.median(random_metrics)),
        "combined_metric_p05": float(np.quantile(random_metrics, 0.05)),
        "combined_metric_p95": float(np.quantile(random_metrics, 0.95)),
        "instrument_mae_mean": float(np.mean(random_instrument_mae)),
        "phase_mae_mean": float(np.mean(random_phase_mae)),
    }

    custom_metric = float(custom_eval["combined_metric"])
    length_metric = float(length_eval["combined_metric"])

    custom_better_than_random_pct = float(np.mean(random_metrics > custom_metric) * 100.0)
    length_better_than_random_pct = float(np.mean(random_metrics > length_metric) * 100.0)

    results_payload = {
        "config": {
            "frames_dir": str(frames_dir),
            "instrument_dir": str(instrument_dir),
            "phase_dir": str(phase_dir),
            "n_videos": len(records),
            "test_size": float(args.test_size),
            "frame_stride": int(args.frame_stride),
            "max_frames_per_video": args.max_frames_per_video,
            "batch_size": int(args.batch_size),
            "convnext_model": args.convnext_model,
            "convnext_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "n_clusters": int(args.n_clusters),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "kmeans_tol": float(args.kmeans_tol),
            "custom_search_trials": int(args.custom_search_trials),
            "length_bins": int(args.length_bins),
            "random_trials": int(args.random_trials),
            "seed": int(args.seed),
        },
        "clustering": {
            "objective_mean_cosine_distance": float(cluster_objective),
            "cluster_counts": {str(k): int(v) for k, v in sorted(cluster_counts.items())},
            "video_cluster": {video_ids[i]: int(cluster_labels[i]) for i in range(len(video_ids))},
        },
        "sampled_frames_per_video": {vid: int(sampled_frames.get(vid, 0)) for vid in video_ids},
        "custom_split": make_split_payload(
            strategy_name="custom_spherical_kmeans_stratified",
            video_ids=video_ids,
            train_idx=custom_train_idx,
            test_idx=custom_test_idx,
            evaluation=custom_eval,
            instrument_labels=instrument_labels,
            phase_labels=phase_labels,
        ),
        "length_split": make_split_payload(
            strategy_name="length_stratified",
            video_ids=video_ids,
            train_idx=length_train_idx,
            test_idx=length_test_idx,
            evaluation=length_eval,
            instrument_labels=instrument_labels,
            phase_labels=phase_labels,
        ),
        "random_baseline": {
            "summary": random_summary,
            "custom_metric": custom_metric,
            "length_metric": length_metric,
            "custom_better_than_random_pct": custom_better_than_random_pct,
            "length_better_than_random_pct": length_better_than_random_pct,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "split_comparison.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    custom_split_path = output_dir / "custom_split_videos.json"
    with custom_split_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_videos": results_payload["custom_split"]["train_videos"],
                "test_videos": results_payload["custom_split"]["test_videos"],
            },
            f,
            indent=2,
        )

    length_split_path = output_dir / "length_split_videos.json"
    with length_split_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_videos": results_payload["length_split"]["train_videos"],
                "test_videos": results_payload["length_split"]["test_videos"],
            },
            f,
            indent=2,
        )

    random_metrics_path = output_dir / "random_metrics.npy"
    np.save(random_metrics_path, random_metrics)

    print("\n=== Split Comparison Summary ===")
    print(f"Custom split combined metric: {custom_metric:.6f}")
    print(f"Length split combined metric: {length_metric:.6f}")
    print(
        "Random split combined metric (mean +- std): "
        f"{random_summary['combined_metric_mean']:.6f} +- {random_summary['combined_metric_std']:.6f}"
    )
    print(f"Custom better than random: {custom_better_than_random_pct:.2f}% of random trials")
    print(f"Length better than random: {length_better_than_random_pct:.2f}% of random trials")
    print(f"Saved full report: {results_path}")
    print(f"Saved custom split videos: {custom_split_path}")
    print(f"Saved length split videos: {length_split_path}")
    print(f"Saved random metric samples: {random_metrics_path}")


if __name__ == "__main__":
    main()
