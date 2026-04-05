"""
Cross-video phase retrieval evaluation.

For each feature extractor: extract embeddings for all phase-labeled videos,
then for every frame find the nearest neighbor in other videos and check
whether it has the same surgical phase.

Usage:
    python src/datasplitting/evaluate.py --models dinov2-base surgenet-dinov2-vits
    python src/datasplitting/evaluate.py --models dinov2-base --compress-dim 384
    python src/datasplitting/evaluate.py --list
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT, _find_existing, _FRAMES_CANDIDATES
from src.feature_extractor import get_extractor, list_extractors
from src.feature_extractor.extract import extract_in_memory
from src.data.read_phases import (
    map_phases_to_videos, match_phases_to_frames, match_video_to_dir,
    discover_available_keys,
)


# ---------------------------------------------------------------------------
# Phase → frame mapping
# ---------------------------------------------------------------------------

def build_frame_phases(
    phases: list[dict],
    n_frames: int,
    all_codes: list[str],
) -> np.ndarray:
    """Map phase timestamps to per-frame phase indices at 1 fps.

    Args:
        phases: list of {label, code, start_ms, end_ms} from read_phases.
        n_frames: number of extracted frames for this video.
        all_codes: global sorted list of phase codes for consistent indexing.

    Returns:
        int32 array of shape (n_frames,). -1 = no phase label.
    """
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    frame_phases = np.full(n_frames, -1, dtype=np.int32)

    for phase in phases:
        start = int(phase["start_ms"] // 1000)
        end = int(phase["end_ms"] // 1000)
        idx = code_to_idx.get(phase["code"], -1)
        if idx < 0:
            continue
        lo = max(0, start)
        hi = min(n_frames, end + 1)
        frame_phases[lo:hi] = idx

    return frame_phases


# ---------------------------------------------------------------------------
# Cross-video nearest-neighbour retrieval
# ---------------------------------------------------------------------------

def _surgery_type(video_key: str) -> str:
    """Extract surgery type from video key (top-level folder)."""
    if "/" in video_key:
        return video_key.split("/", 1)[0]
    return video_key


def cross_video_retrieval(
    video_embs: dict[str, np.ndarray],
    video_phases: dict[str, np.ndarray],
    beta: float = 0.0,
) -> dict:
    """For each frame, find nearest neighbour in all other videos.

    If beta > 0, only count a retrieval if cosine_sim >= next_frame_sim / beta.
    This filters out matches that are too dissimilar relative to temporal neighbours.

    Returns dict with phase accuracy, surgery type accuracy, and breakdowns.
    """
    video_keys = sorted(video_embs.keys())

    # Surgery type per video
    vid_surgery = [_surgery_type(vk) for vk in video_keys]

    # Collect only phase-labeled frames (so every query always finds a match)
    all_embs = []
    all_vid_idx = []
    all_phase = []

    for vi, vkey in enumerate(video_keys):
        emb = video_embs[vkey].astype(np.float32)
        phases = video_phases[vkey]
        n = min(len(emb), len(phases))
        for fi in range(n):
            if phases[fi] >= 0:
                all_embs.append(emb[fi])
                all_vid_idx.append(vi)
                all_phase.append(phases[fi])

    all_embs = np.vstack(all_embs).astype(np.float32)
    all_vid_idx = np.array(all_vid_idx, dtype=np.int64)
    all_phase = np.array(all_phase, dtype=np.int32)

    # L2 normalise → cosine similarity via inner product
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embs = (all_embs / norms).astype(np.float32)

    # Pre-compute per-frame next-frame similarity (within same video) for beta filter
    # next_sim[i] = cosine_sim(frame_i, frame_i+1) within same video, or NaN for last frame
    next_sim = np.full(len(all_embs), np.nan, dtype=np.float32)
    if beta > 0:
        for vi in range(max(all_vid_idx) + 1):
            mask = np.where(all_vid_idx == vi)[0]
            if len(mask) < 2:
                continue
            for j in range(len(mask) - 1):
                next_sim[mask[j]] = float(np.dot(all_embs[mask[j]], all_embs[mask[j + 1]]))

    # Build FAISS index (flat inner-product) — only labeled frames
    import faiss

    d = all_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_embs)

    # Per-video search: build per-video index excluding same-video frames, k=1
    phase_correct = 0
    stype_correct = 0
    total = 0
    skipped = 0
    per_vid_phase_correct = np.zeros(len(video_keys), dtype=np.int64)
    per_vid_stype_correct = np.zeros(len(video_keys), dtype=np.int64)
    per_vid_total = np.zeros(len(video_keys), dtype=np.int64)
    per_phase_correct: dict[int, int] = {}
    per_phase_total: dict[int, int] = {}
    per_stype_correct: dict[str, int] = {}
    per_stype_total: dict[str, int] = {}

    for vi in range(len(video_keys)):
        q_mask = all_vid_idx == vi
        if not q_mask.any():
            continue

        q_global = np.where(q_mask)[0]  # global indices of query frames
        q_embs = all_embs[q_mask]
        q_phases = all_phase[q_mask]

        # Build index from all OTHER videos' frames
        other_mask = ~q_mask
        other_embs = all_embs[other_mask]
        other_phase = all_phase[other_mask]
        other_vid_idx = all_vid_idx[other_mask]

        other_index = faiss.IndexFlatIP(d)
        other_index.add(other_embs)

        # k=1: best match from other videos
        D, I = other_index.search(q_embs, 1)

        q_stype = vid_surgery[vi]
        for qi_local in range(len(q_embs)):
            nn = I[qi_local, 0]
            if nn < 0:
                continue

            # Beta filter: cross_sim must be >= next_sim / beta
            if beta > 0:
                ns = next_sim[q_global[qi_local]]
                if not np.isnan(ns) and D[qi_local, 0] < ns / beta:
                    skipped += 1
                    continue

            nn_phase = other_phase[nn]
            nn_stype = vid_surgery[other_vid_idx[nn]]
            q_phase = q_phases[qi_local]

            p_match = int(nn_phase == q_phase)
            phase_correct += p_match
            per_vid_phase_correct[vi] += p_match
            per_phase_correct[q_phase] = per_phase_correct.get(q_phase, 0) + p_match
            per_phase_total[q_phase] = per_phase_total.get(q_phase, 0) + 1

            s_match = int(nn_stype == q_stype)
            stype_correct += s_match
            per_vid_stype_correct[vi] += s_match
            per_stype_correct[q_stype] = per_stype_correct.get(q_stype, 0) + s_match
            per_stype_total[q_stype] = per_stype_total.get(q_stype, 0) + 1

            total += 1
            per_vid_total[vi] += 1

    phase_accuracy = phase_correct / total if total > 0 else 0.0
    stype_accuracy = stype_correct / total if total > 0 else 0.0

    per_video_phase_acc = {}
    per_video_stype_acc = {}
    for vi, vkey in enumerate(video_keys):
        t = int(per_vid_total[vi])
        per_video_phase_acc[vkey] = int(per_vid_phase_correct[vi]) / t if t > 0 else 0.0
        per_video_stype_acc[vkey] = int(per_vid_stype_correct[vi]) / t if t > 0 else 0.0

    per_phase_acc = {}
    for pi in sorted(per_phase_total):
        t = per_phase_total[pi]
        c = per_phase_correct.get(pi, 0)
        per_phase_acc[pi] = c / t if t > 0 else 0.0

    per_stype_acc = {}
    for st in sorted(per_stype_total):
        t = per_stype_total[st]
        c = per_stype_correct.get(st, 0)
        per_stype_acc[st] = c / t if t > 0 else 0.0

    return {
        "phase_accuracy": phase_accuracy,
        "phase_correct": phase_correct,
        "stype_accuracy": stype_accuracy,
        "stype_correct": stype_correct,
        "total": total,
        "skipped_by_beta": skipped,
        "per_video_phase_accuracy": per_video_phase_acc,
        "per_video_stype_accuracy": per_video_stype_acc,
        "per_phase_accuracy": per_phase_acc,
        "per_stype_accuracy": per_stype_acc,
    }


# ---------------------------------------------------------------------------
# PCA dimension reduction (reused from old code)
# ---------------------------------------------------------------------------

def reduce_dim(
    video_embs: dict[str, np.ndarray],
    target_dim: int,
) -> dict[str, np.ndarray]:
    """Reduce embedding dimension via PCA."""
    from sklearn.decomposition import PCA

    all_feats = np.vstack([v.astype(np.float32) for v in video_embs.values()])
    current_dim = all_feats.shape[1]
    if target_dim >= current_dim:
        print(f"  Skipping PCA: target_dim={target_dim} >= current_dim={current_dim}")
        return video_embs

    print(f"  PCA: {current_dim} -> {target_dim} (fitted on {len(all_feats):,} frames)")
    pca = PCA(n_components=target_dim, random_state=42)
    pca.fit(all_feats)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return {
        vid: pca.transform(emb.astype(np.float32)).astype(np.float16)
        for vid, emb in video_embs.items()
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _load_embeddings_from_dir(
    embeddings_dir: Path,
    video_filter: set[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load pre-computed .npy embeddings from directory structure.

    Expects: embeddings_dir/<category>/<video_name>/*.npy
    or:      embeddings_dir/<video_name>/*.npy
    """
    video_embs: dict[str, np.ndarray] = {}
    cat_dirs = sorted(d for d in embeddings_dir.iterdir() if d.is_dir())

    for cat_dir in cat_dirs:
        if categories and cat_dir.name not in categories:
            continue
        # Check if cat_dir itself has .npy files (flat layout)
        npys = sorted(cat_dir.glob("*.npy"))
        if npys:
            key = cat_dir.name
            if video_filter and key not in video_filter:
                continue
            feats = [np.load(f) for f in npys]
            video_embs[key] = np.vstack(feats) if feats[0].ndim == 1 else np.concatenate(feats)
        else:
            # Nested: category/video/*.npy
            for vid_dir in sorted(cat_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                key = f"{cat_dir.name}/{vid_dir.name}"
                if video_filter and key not in video_filter:
                    continue
                npys = sorted(vid_dir.glob("*.npy"))
                if not npys:
                    continue
                feats = [np.load(f) for f in npys]
                video_embs[key] = np.vstack(feats) if feats[0].ndim == 1 else np.concatenate(feats)

    return video_embs


def run_evaluation(
    models: list[str],
    frames_dir: Path,
    categories: list[str] | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    compress_dim: int | None = None,
    device: str = "cuda:0",
    embeddings_dir: Path | None = None,
    beta: float = 0.0,
) -> dict[str, dict]:
    """Run cross-video phase retrieval for each model."""

    # Load phase labels
    phase_data, found, missing = map_phases_to_videos()
    print(f"Phase labels: {found}/{len(phase_data)} videos matched to Excel")
    if missing:
        print(f"  Missing UUIDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Collect all unique phase codes across all videos
    all_codes = sorted({
        p["code"]
        for v in phase_data
        for p in v["phases"]
    })
    all_labels = sorted({
        p["label"]
        for v in phase_data
        for p in v["phases"]
    })
    print(f"Phase codes: {all_codes}")

    # Pre-match: scan directory names (cheap), match phase labels, build filter
    # Use embeddings_dir for matching if provided and no frames_dir
    match_dir = embeddings_dir if embeddings_dir else frames_dir
    video_matches, dir_key_users = match_phases_to_frames(
        phase_data, match_dir, categories,
    )
    matched_keys = set(video_matches.values())

    for vinfo in phase_data:
        if vinfo["name"] is None:
            print(f"  SKIP: unnamed video (UUID {vinfo['video_id'][:16]}...)")
            continue
        dk = video_matches.get(vinfo["name"])
        if dk:
            print(f"  OK:   {vinfo['name']:<30} -> {dk}")
        else:
            print(f"  MISS: {vinfo['name']} ({vinfo.get('dataset')})")

    n_no_name = sum(1 for r in phase_data if r["name"] is None)
    if n_no_name:
        print(f"  WARN: {n_no_name} videos have no name (UUID not found in Excel)")
    for dk, names in dir_key_users.items():
        if len(names) > 1:
            print(f"  WARN: {len(names)} videos map to same dir {dk}:")
            for n in names:
                print(f"         - {n}")
    n_named = sum(1 for r in phase_data if r["name"] is not None)
    n_miss = n_named - len(video_matches)
    print(f"Phase-labeled videos: {len(phase_data)} total, {len(video_matches)} matched to dirs "
          f"({len(matched_keys)} unique), {n_miss} missing, {n_no_name} unnamed")

    if len(matched_keys) < 2:
        print("ABORT: need at least 2 videos with phase labels")
        return {}

    device_obj = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if embeddings_dir:
            print(f"  Loading pre-computed embeddings from {embeddings_dir}")
            video_embs = _load_embeddings_from_dir(
                embeddings_dir, video_filter=matched_keys, categories=categories,
            )
        else:
            extractor = get_extractor(model_name)
            extractor.load_model(device_obj)

            video_embs = extract_in_memory(
                extractor, frames_dir,
                batch_size=batch_size, num_workers=num_workers,
                categories=categories,
                video_filter=matched_keys,
            )
            del extractor
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

        if compress_dim:
            video_embs = reduce_dim(video_embs, compress_dim)

        print(f"  Loaded: {len(video_embs)} videos (only phase-labeled)")

        # Build phase arrays for matched videos
        matched_embs: dict[str, np.ndarray] = {}
        matched_phases: dict[str, np.ndarray] = {}

        for vinfo in phase_data:
            if vinfo["name"] is None or vinfo["name"] not in video_matches:
                continue
            dir_key = video_matches[vinfo["name"]]
            if dir_key not in video_embs:
                continue

            emb = video_embs[dir_key]
            n_frames = len(emb)
            frame_phases = build_frame_phases(vinfo["phases"], n_frames, all_codes)

            labeled = int((frame_phases >= 0).sum())
            print(f"  {vinfo['name']:<30} -> {dir_key:<50} "
                  f"({n_frames} frames, {labeled} labeled)")

            matched_embs[dir_key] = emb
            matched_phases[dir_key] = frame_phases

        print(f"\n  Matched: {len(matched_embs)} videos")

        if len(matched_embs) < 2:
            print("  SKIP: need at least 2 videos with phase labels")
            continue

        # Cross-video retrieval
        betas = [beta] if beta > 0 else [0.0]
        print(f"  Running cross-video NN retrieval (beta={beta})...")
        retrieval = cross_video_retrieval(matched_embs, matched_phases, beta=beta)

        print(f"  Phase accuracy:   {retrieval['phase_accuracy']:.2%} "
              f"({retrieval['phase_correct']}/{retrieval['total']})")
        print(f"  Surgery type acc: {retrieval['stype_accuracy']:.2%} "
              f"({retrieval['stype_correct']}/{retrieval['total']})")
        if beta > 0:
            print(f"  Skipped by beta:  {retrieval['skipped_by_beta']}")

        # Per-phase breakdown
        for pi, acc in sorted(retrieval["per_phase_accuracy"].items()):
            phase_name = all_codes[pi] if pi < len(all_codes) else f"phase_{pi}"
            print(f"    phase  {phase_name}: {acc:.2%}")

        # Per-surgery-type breakdown
        for st, acc in sorted(retrieval["per_stype_accuracy"].items()):
            print(f"    stype  {st}: {acc:.2%}")

        results[model_name] = {
            "model": model_name,
            "phase_accuracy": retrieval["phase_accuracy"],
            "phase_correct": retrieval["phase_correct"],
            "stype_accuracy": retrieval["stype_accuracy"],
            "stype_correct": retrieval["stype_correct"],
            "total": retrieval["total"],
            "skipped_by_beta": retrieval["skipped_by_beta"],
            "beta": beta,
            "n_videos": len(matched_embs),
            "per_video_phase_accuracy": retrieval["per_video_phase_accuracy"],
            "per_video_stype_accuracy": retrieval["per_video_stype_accuracy"],
            "per_phase_accuracy": {
                all_codes[pi] if pi < len(all_codes) else f"phase_{pi}": acc
                for pi, acc in retrieval["per_phase_accuracy"].items()
            },
            "per_stype_accuracy": retrieval["per_stype_accuracy"],
            "phase_labels": all_labels,
            "compress_dim": compress_dim,
        }

    return results


def print_comparison(results: dict[str, dict]) -> None:
    """Print a comparison table of all models."""
    print(f"\n{'='*85}")
    print("COMPARISON — Cross-Video Retrieval")
    print(f"{'='*85}")
    print(f"{'Model':<35} {'Dim':>5} {'Phase Acc':>10} {'SType Acc':>10} {'Total':>8} {'Vids':>5}")
    print("-" * 85)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["phase_accuracy"]):
        dim_str = str(r.get("compress_dim") or "orig")
        print(f"{name:<35} {dim_str:>5} {r['phase_accuracy']:>9.2%} "
              f"{r['stype_accuracy']:>9.2%} {r['total']:>8} {r['n_videos']:>5}")

    # Per-phase comparison if multiple models
    if len(results) > 1:
        all_phases = set()
        for r in results.values():
            all_phases.update(r.get("per_phase_accuracy", {}).keys())
        if all_phases:
            print(f"\nPer-phase accuracy:")
            header = f"{'Model':<35}" + "".join(f" {p:>18}" for p in sorted(all_phases))
            print(header)
            print("-" * len(header))
            for name, r in sorted(results.items(), key=lambda x: -x[1]["phase_accuracy"]):
                pa = r.get("per_phase_accuracy", {})
                cols = "".join(
                    f" {pa.get(p, 0):.2%}".rjust(18)
                    for p in sorted(all_phases)
                )
                print(f"{name:<35}{cols}")

        # Per-surgery-type comparison
        all_stypes = set()
        for r in results.values():
            all_stypes.update(r.get("per_stype_accuracy", {}).keys())
        if all_stypes:
            print(f"\nPer-surgery-type accuracy:")
            header = f"{'Model':<35}" + "".join(f" {s:>18}" for s in sorted(all_stypes))
            print(header)
            print("-" * len(header))
            for name, r in sorted(results.items(), key=lambda x: -x[1]["stype_accuracy"]):
                sa = r.get("per_stype_accuracy", {})
                cols = "".join(
                    f" {sa.get(s, 0):.2%}".rjust(18)
                    for s in sorted(all_stypes)
                )
                print(f"{name:<35}{cols}")


def run_beta_sweep(
    betas: list[float],
    models: list[str],
    frames_dir: Path,
    embeddings_dir: Path | None = None,
    categories: list[str] | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    compress_dim: int | None = None,
    device: str = "cuda:0",
) -> None:
    """Run retrieval once (building embeddings), then sweep beta values."""

    # Load phase labels
    phase_data, found, missing = map_phases_to_videos()
    print(f"Phase labels: {found}/{len(phase_data)} videos matched to Excel")

    all_codes = sorted({p["code"] for v in phase_data for p in v["phases"]})
    print(f"Phase codes: {all_codes}")

    match_dir = embeddings_dir if embeddings_dir else frames_dir
    video_matches, _ = match_phases_to_frames(phase_data, match_dir, categories)
    matched_keys = set(video_matches.values())

    if len(matched_keys) < 2:
        print("ABORT: need at least 2 videos with phase labels")
        return

    device_obj = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if embeddings_dir:
            print(f"  Loading pre-computed embeddings from {embeddings_dir}")
            video_embs = _load_embeddings_from_dir(
                embeddings_dir, video_filter=matched_keys, categories=categories,
            )
        else:
            extractor = get_extractor(model_name)
            extractor.load_model(device_obj)
            video_embs = extract_in_memory(
                extractor, frames_dir,
                batch_size=batch_size, num_workers=num_workers,
                categories=categories, video_filter=matched_keys,
            )
            del extractor
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

        if compress_dim:
            video_embs = reduce_dim(video_embs, compress_dim)

        # Build phase arrays
        matched_embs: dict[str, np.ndarray] = {}
        matched_phases: dict[str, np.ndarray] = {}
        for vinfo in phase_data:
            if vinfo["name"] is None or vinfo["name"] not in video_matches:
                continue
            dir_key = video_matches[vinfo["name"]]
            if dir_key not in video_embs:
                continue
            emb = video_embs[dir_key]
            matched_embs[dir_key] = emb
            matched_phases[dir_key] = build_frame_phases(vinfo["phases"], len(emb), all_codes)

        print(f"  {len(matched_embs)} videos loaded\n")

        if len(matched_embs) < 2:
            print("  SKIP: need >= 2 videos")
            continue

        # Header
        phase_names = all_codes
        print(f"{'beta':>6} | {'phase_acc':>9} {'stype_acc':>9} {'matched':>8} {'skipped':>8} {'pct_matched':>11} | "
              + " ".join(f"{p:>12}" for p in phase_names)
              + " | " + "per-stype")
        print("-" * 140)

        for b in betas:
            r = cross_video_retrieval(matched_embs, matched_phases, beta=b)
            total_possible = r["total"] + r["skipped_by_beta"]
            pct = r["total"] / total_possible * 100 if total_possible > 0 else 0
            phase_cols = " ".join(
                f"{r['per_phase_accuracy'].get(pi, 0):.2%}".rjust(12)
                for pi in range(len(phase_names))
            )
            stype_parts = ", ".join(
                f"{st}={acc:.2%}" for st, acc in sorted(r["per_stype_accuracy"].items())
            )
            print(f"{b:>6.2f} | {r['phase_accuracy']:>8.2%} {r['stype_accuracy']:>9.2%} "
                  f"{r['total']:>8} {r['skipped_by_beta']:>8} {pct:>10.1f}% | "
                  f"{phase_cols} | {stype_parts}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-video phase retrieval evaluation"
    )
    parser.add_argument("--models", nargs="*", default=["surgenet-dinov2-vits"])
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--frames-dir", type=Path,
                        default=_find_existing(_FRAMES_CANDIDATES, _FRAMES_CANDIDATES[0]))
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="Load pre-computed .npy embeddings instead of extracting")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compress-dim", type=int, default=None,
                        help="Reduce embeddings to this dimension via PCA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Min similarity threshold: cross_sim >= next_sim / beta")
    parser.add_argument("--beta-sweep", nargs="*", type=float, default=None,
                        help="Run sweep over multiple beta values")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in list_extractors():
            print(f"  {name}")
        return

    if args.beta_sweep:
        run_beta_sweep(
            betas=sorted(args.beta_sweep, reverse=True),
            models=args.models,
            frames_dir=args.frames_dir,
            embeddings_dir=args.embeddings_dir,
            categories=args.categories,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            compress_dim=args.compress_dim,
            device=args.device,
        )
        return

    results = run_evaluation(
        models=args.models,
        frames_dir=args.frames_dir,
        categories=args.categories,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compress_dim=args.compress_dim,
        device=args.device,
        embeddings_dir=args.embeddings_dir,
        beta=args.beta,
    )

    print_comparison(results)

    # Save results
    out_file = PROJECT_ROOT / "output" / "datasplitting" / "retrieval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
