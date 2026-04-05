"""
Offline preprocessing: temporal discontinuity scoring and cross-video pair mining.

Absorbs core algorithms from own experiments/sampler/ scripts.
Outputs JSON files consumed by the dataset and tools.

Usage:
    python -m src.data.preprocessing temporal --embeddings-root /path/to/embeddings --output temporal_scores.json
    python -m src.data.preprocessing pairs --embeddings-root /path/to/embeddings --factor 3.0 --output pair_index.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_feature_paths_by_video(
    features_root: Path,
    exclude_folders: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Group .npy feature files by parent folder (= video name)."""
    all_features = sorted(features_root.rglob("*.npy"))
    videos: dict[str, list[Path]] = defaultdict(list)
    for path in all_features:
        videos[path.parent.name].append(path)
    for name in list(videos):
        videos[name] = sorted(videos[name])
    if exclude_folders:
        for folder in exclude_folders:
            if folder in videos:
                del videos[folder]
                print(f"[preprocessing] Excluded: {folder}")
    return dict(videos)


def load_video_features(feature_paths: list[Path]) -> np.ndarray | None:
    """Load all .npy features for one video, interpolating failures."""
    features: list[np.ndarray | None] = []
    for p in feature_paths:
        try:
            features.append(np.load(p))
        except Exception as e:
            print(f"[preprocessing] Error loading {p}: {e}")
            features.append(None)
    valid = [i for i, f in enumerate(features) if f is not None]
    if not valid:
        return None
    for i, f in enumerate(features):
        if f is None:
            nearest = min(valid, key=lambda x: abs(x - i))
            features[i] = features[nearest]
    return np.vstack(features)


def feature_path_to_image_path(
    feature_path: Path,
    features_root: Path,
    images_root: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> Path | None:
    """Convert .npy feature path to its source image path."""
    relative = feature_path.relative_to(features_root)
    stem_path = images_root / relative.with_suffix("")
    for ext in extensions:
        for e in (ext, ext.upper()):
            candidate = stem_path.with_suffix(e)
            if candidate.exists():
                return candidate
    return stem_path.with_suffix(".jpg")


# ---------------------------------------------------------------------------
# Temporal discontinuity detection
# ---------------------------------------------------------------------------

@dataclass
class TemporalChange:
    position: int
    change_score: float
    video_name: str
    window_size: int


def compute_temporal_changes(features: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Per-position cosine distance between mean(before_window) and mean(after_window).
    Returns array of length (n_frames - 2*window_size + 1).
    """
    n = len(features)
    if n < 2 * window_size:
        return np.array([])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feat = features / norms
    scores = []
    for i in range(window_size, n - window_size + 1):
        prev = feat[i - window_size:i].mean(axis=0)
        prev /= np.linalg.norm(prev) + 1e-8
        nxt = feat[i:i + window_size].mean(axis=0)
        nxt /= np.linalg.norm(nxt) + 1e-8
        scores.append(1 - float(np.dot(prev, nxt)))
    return np.array(scores)


def compute_all_temporal_scores(
    features_root: Path,
    window_size: int = 120,
    min_gap: int = 30,
    exclude_folders: list[str] | None = None,
) -> tuple[list[TemporalChange], np.ndarray]:
    """
    Compute temporal change scores across all videos.
    Returns (local_maxima_changes, all_raw_scores).
    """
    videos = get_feature_paths_by_video(features_root, exclude_folders)
    print(f"[temporal] Processing {len(videos)} videos, window={window_size}")
    all_changes: list[TemporalChange] = []
    all_scores: list[float] = []

    for video_name, paths in tqdm(videos.items(), desc="Temporal scores"):
        if len(paths) < 2 * window_size:
            continue
        features = load_video_features(paths)
        if features is None:
            continue
        scores = compute_temporal_changes(features, window_size)
        if len(scores) == 0:
            continue
        all_scores.extend(scores.tolist())
        # Local maxima with min_gap
        order = np.argsort(scores)[::-1]
        selected: list[int] = []
        for pos in order:
            actual = pos + window_size
            if not any(abs(actual - s) < min_gap for s in selected):
                selected.append(actual)
                all_changes.append(TemporalChange(
                    position=actual,
                    change_score=float(scores[pos]),
                    video_name=video_name,
                    window_size=window_size,
                ))
    all_changes.sort(key=lambda c: c.change_score, reverse=True)
    return all_changes, np.array(all_scores)


def save_temporal_scores(
    changes: list[TemporalChange],
    all_scores: np.ndarray,
    output_path: Path,
):
    """Save temporal scores to JSON for tools/training consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "changes": [
            {"video": c.video_name, "position": c.position,
             "score": c.change_score, "window_size": c.window_size}
            for c in changes
        ],
        "stats": {
            "n_scores": len(all_scores),
            "min": float(all_scores.min()) if len(all_scores) else 0,
            "max": float(all_scores.max()) if len(all_scores) else 0,
            "mean": float(all_scores.mean()) if len(all_scores) else 0,
            "median": float(np.median(all_scores)) if len(all_scores) else 0,
            "percentiles": {
                str(p): float(np.percentile(all_scores, p))
                for p in (50, 90, 95, 99, 99.5)
            } if len(all_scores) else {},
        },
        "all_scores": all_scores.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"[temporal] Saved {len(changes)} changes + {len(all_scores)} scores → {output_path}")


# ---------------------------------------------------------------------------
# Cross-video pair mining (FAISS-based)
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    query_path: str
    query_video: str
    next_frame_similarity: float | None
    best_cross_video_path: str | None
    best_cross_video: str | None
    cross_video_similarity: float | None
    is_match: bool


def _load_all_features(
    videos: dict[str, list[Path]],
) -> tuple[np.ndarray, list[Path], list[str]]:
    """Stack all embeddings into one array."""
    feats, paths, labels = [], [], []
    for video_name, vpaths in tqdm(videos.items(), desc="Loading features"):
        for p in vpaths:
            try:
                feats.append(np.load(p))
                paths.append(p)
                labels.append(video_name)
            except Exception as e:
                print(f"[pairs] Error loading {p}: {e}")
    return np.vstack(feats).astype(np.float32), paths, labels


def build_pair_index(
    features_root: Path,
    factor: float = 3.0,
    top_k: int = 5,
    n_samples: int | None = None,
    exclude_folders: list[str] | None = None,
    seed: int = 42,
) -> tuple[list[MatchResult], dict]:
    """
    Cross-video pair mining using FAISS for scalability.

    For each frame (or a random sample), find the nearest neighbor from a
    different video. Keep the pair if cross_sim > next_frame_sim / factor.

    Args:
        features_root: directory with video_name/*.npy
        factor: LemonFM threshold factor (beta)
        top_k: number of FAISS neighbors to search
        n_samples: if set, only sample this many frames (for UI preview)
        exclude_folders: folders to skip
        seed: random seed

    Returns:
        (results, pair_dict) where pair_dict maps frame_path → [matched_paths]
    """
    random.seed(seed)
    np.random.seed(seed)
    videos = get_feature_paths_by_video(features_root, exclude_folders)
    if len(videos) < 2:
        print("[pairs] Need at least 2 videos!")
        return [], {}

    print(f"[pairs] {len(videos)} videos, factor={factor}, top_k={top_k}")
    all_features, all_paths, all_labels = _load_all_features(videos)
    n_total = len(all_features)
    print(f"[pairs] Total embeddings: {n_total}, dim={all_features.shape[1]}")

    # Video index lookup
    video_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(all_labels):
        video_indices[label].append(idx)

    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(all_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_features_norm = all_features / norms

    # Try FAISS, fall back to sklearn
    try:
        import faiss
        dim = all_features_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_features_norm)
        use_faiss = True
        print(f"[pairs] Using FAISS IndexFlatIP (dim={dim})")
    except ImportError:
        from sklearn.metrics.pairwise import cosine_similarity
        use_faiss = False
        print("[pairs] FAISS not available, falling back to sklearn")

    # Determine which frames to process
    if n_samples is not None and n_samples < n_total:
        valid_videos = [v for v, ps in videos.items() if len(ps) > 1]
        sample_indices = []
        for _ in range(n_samples):
            v = random.choice(valid_videos)
            vil = video_indices[v]
            sample_indices.append(vil[random.randint(0, len(vil) - 2)])
    else:
        sample_indices = list(range(n_total))

    results: list[MatchResult] = []
    pair_dict: dict[str, list[str]] = {}

    for qi in tqdm(sample_indices, desc="Mining pairs"):
        query_video = all_labels[qi]
        vil = video_indices[query_video]
        local_idx = vil.index(qi)

        # Next-frame similarity
        if local_idx + 1 < len(vil):
            nfi = vil[local_idx + 1]
            next_sim = float(np.dot(all_features_norm[qi], all_features_norm[nfi]))
        else:
            next_sim = None

        # Find cross-video nearest neighbor
        if use_faiss:
            dists, idxs = index.search(all_features_norm[qi:qi + 1], top_k + 10)
            best_cross_sim, best_cross_idx = -1.0, None
            for sim, idx in zip(dists[0], idxs[0]):
                if idx >= 0 and all_labels[idx] != query_video and sim > best_cross_sim:
                    best_cross_sim = float(sim)
                    best_cross_idx = int(idx)
                    break
        else:
            sims = cosine_similarity(all_features_norm[qi:qi + 1], all_features_norm)[0]
            best_cross_sim, best_cross_idx = -1.0, None
            for j, (s, lab) in enumerate(zip(sims, all_labels)):
                if lab != query_video and s > best_cross_sim:
                    best_cross_sim = float(s)
                    best_cross_idx = j

        # Apply LemonFM criterion
        if next_sim is not None and best_cross_idx is not None:
            is_match = best_cross_sim > (next_sim / factor)
        else:
            is_match = False

        qp = str(all_paths[qi])
        bcp = str(all_paths[best_cross_idx]) if best_cross_idx is not None else None
        bcv = all_labels[best_cross_idx] if best_cross_idx is not None else None

        results.append(MatchResult(
            query_path=qp, query_video=query_video,
            next_frame_similarity=next_sim,
            best_cross_video_path=bcp, best_cross_video=bcv,
            cross_video_similarity=best_cross_sim if best_cross_idx else None,
            is_match=is_match,
        ))

        if is_match and bcp:
            pair_dict.setdefault(qp, []).append(bcp)

    return results, pair_dict


def save_pair_index(pair_dict: dict, results: list[MatchResult], output_path: Path, factor: float):
    """Save cross-video pair index to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_match = sum(1 for r in results if r.is_match)
    data = {
        "factor": factor,
        "n_total": len(results),
        "n_matches": n_match,
        "match_rate": n_match / len(results) if results else 0,
        "pairs": pair_dict,
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"[pairs] Saved {len(pair_dict)} pairs ({n_match}/{len(results)} matches) → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline preprocessing for foundation model pretraining")
    sub = parser.add_subparsers(dest="command")

    # Temporal
    tp = sub.add_parser("temporal", help="Compute temporal discontinuity scores")
    tp.add_argument("--embeddings-root", type=Path, required=True)
    tp.add_argument("--output", type=Path, default=Path("output/temporal_scores.json"))
    tp.add_argument("--window-size", type=int, default=120)
    tp.add_argument("--min-gap", type=int, default=30)
    tp.add_argument("--exclude", nargs="*", default=["reference for filtering"])

    # Pairs
    pp = sub.add_parser("pairs", help="Cross-video pair mining")
    pp.add_argument("--embeddings-root", type=Path, required=True)
    pp.add_argument("--output", type=Path, default=Path("output/pair_index.json"))
    pp.add_argument("--factor", type=float, default=3.0)
    pp.add_argument("--top-k", type=int, default=5)
    pp.add_argument("--exclude", nargs="*", default=["reference for filtering"])

    args = parser.parse_args()

    if args.command == "temporal":
        changes, scores = compute_all_temporal_scores(
            args.embeddings_root, args.window_size, args.min_gap, args.exclude)
        save_temporal_scores(changes, scores, args.output)
    elif args.command == "pairs":
        results, pair_dict = build_pair_index(
            args.embeddings_root, args.factor, args.top_k,
            exclude_folders=args.exclude)
        save_pair_index(pair_dict, results, args.output, args.factor)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
