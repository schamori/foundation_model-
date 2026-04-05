"""
CLI for feature extraction with any registered model.

Usage:
    python src/feature_extractor/extract.py --model dinov2-base
    python src/feature_extractor/extract.py --model surgenet-dinov3-vitl --batch-size 64
    python src/feature_extractor/extract.py --list
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

# Support running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.feature_extractor import get_extractor, list_extractors, FeatureExtractor
from src.config import _find_existing, _FRAMES_CANDIDATES, _EMBEDDINGS_CANDIDATES, PROJECT_ROOT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class _FrameDataset(Dataset):
    def __init__(self, records: list[dict], extractor: FeatureExtractor):
        self.records = records
        self.extractor = extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            with Image.open(rec["frame_path"]) as img:
                pv = self.extractor.preprocess([img.convert("RGB")])[0]
            return pv, rec["output_path"], True
        except Exception:
            return torch.zeros(3, self.extractor.input_size, self.extractor.input_size), rec["output_path"], False


def _collate(batch):
    pv, paths, ok = zip(*batch)
    return torch.stack(pv), list(paths), list(ok)


def extract_all(
    extractor: FeatureExtractor,
    frames_dir: Path,
    output_dir: Path,
    batch_size: int = 128,
    num_workers: int = 8,
    force: bool = False,
    categories: list[str] | None = None,
):
    """Extract features for all frames under frames_dir.

    Directory structure: frames_dir/[category/]video/frame.jpg
    Output mirrors: output_dir/[category/]video/frame.npy
    """
    # Discover frames — handle both flat and nested layouts
    records = []
    for item in sorted(frames_dir.iterdir()):
        if not item.is_dir():
            continue
        # Check if this is a category dir (contains subdirs) or video dir (contains images)
        has_images = any(f.suffix.lower() in IMAGE_SUFFIXES for f in item.iterdir() if f.is_file())
        if has_images:
            # Flat: frames_dir/video/frame.jpg
            _add_video_records(records, item, output_dir / item.name)
        else:
            # Nested: frames_dir/category/video/frame.jpg
            if categories and item.name not in categories:
                continue
            for vdir in sorted(item.iterdir()):
                if vdir.is_dir():
                    _add_video_records(records, vdir, output_dir / item.name / vdir.name)

    if not records:
        print("No frames found.")
        return

    if not force:
        todo = [r for r in records if not Path(r["output_path"]).exists()]
    else:
        todo = records

    print(f"Frames: {len(todo):,} to extract / {len(records):,} total")
    if not todo:
        print("All already extracted.")
        return

    loader = DataLoader(
        _FrameDataset(todo, extractor),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
    )

    saved = failed = 0
    for pv, out_paths, valids in tqdm(loader, desc=f"Extracting [{extractor.name}]"):
        feats = extractor.extract_features(pv)
        for feat, op, valid in zip(feats, out_paths, valids):
            if not valid:
                failed += 1
                continue
            p = Path(op)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, feat)
            saved += 1

    print(f"Saved {saved:,} embeddings ({failed} failed)")


def _discover_frames(
    frames_dir: Path,
    categories: list[str] | None = None,
    video_filter: set[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover frames grouped by video key (e.g. 'MVD/MVD_001' or 'video01').

    Args:
        video_filter: if provided, only include video keys in this set.

    Returns {video_key: [frame_paths]}.
    """
    videos: dict[str, list[Path]] = {}
    for item in sorted(frames_dir.iterdir()):
        if not item.is_dir():
            continue
        has_images = any(f.suffix.lower() in IMAGE_SUFFIXES for f in item.iterdir() if f.is_file())
        if has_images:
            if video_filter and item.name not in video_filter:
                continue
            paths = sorted(f for f in item.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            if paths:
                videos[item.name] = paths
        else:
            if categories and item.name not in categories:
                continue
            for vdir in sorted(item.iterdir()):
                if vdir.is_dir():
                    key = f"{item.name}/{vdir.name}"
                    if video_filter and key not in video_filter:
                        continue
                    paths = sorted(f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
                    if paths:
                        videos[key] = paths
    return videos


class _InMemoryFrameDataset(Dataset):
    """Dataset that returns (pixel_values, video_key, valid) — no output path."""
    def __init__(self, frame_paths: list[Path], video_keys: list[str], extractor: FeatureExtractor):
        self.frame_paths = frame_paths
        self.video_keys = video_keys
        self.extractor = extractor

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        with Image.open(self.frame_paths[idx]) as img:
            pv = self.extractor.preprocess([img.convert("RGB")])[0]
        return pv, self.video_keys[idx], True



def _collate_in_memory(batch):
    pv, keys, ok = zip(*batch)
    return torch.stack(pv), list(keys), list(ok)


def extract_in_memory(
    extractor: FeatureExtractor,
    frames_dir: Path,
    batch_size: int = 128,
    num_workers: int = 8,
    categories: list[str] | None = None,
    video_filter: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract features in memory without saving to disk.

    Args:
        video_filter: if provided, only extract these video keys.

    Returns {video_key: (N, D) float16 array}.
    """
    from collections import defaultdict

    videos = _discover_frames(frames_dir, categories, video_filter)
    if not videos:
        print("No frames found.")
        return {}

    # Flatten for DataLoader
    all_paths, all_keys = [], []
    for vkey, paths in videos.items():
        for p in paths:
            all_paths.append(p)
            all_keys.append(vkey)

    total = len(all_paths)
    print(f"Extracting {total:,} frames in memory (no disk writes)")

    loader = DataLoader(
        _InMemoryFrameDataset(all_paths, all_keys, extractor),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate_in_memory,
    )

    video_feats: dict[str, list[np.ndarray]] = defaultdict(list)
    failed = 0
    for pv, keys, valids in tqdm(loader, desc=f"Extracting [{extractor.name}] (in-memory)"):
        feats = extractor.extract_features(pv)
        for feat, vkey, valid in zip(feats, keys, valids):
            if valid:
                video_feats[vkey].append(feat)
            else:
                failed += 1

    if failed:
        print(f"  {failed} frames failed")

    # Stack per-video
    result = {}
    for vkey, feat_list in video_feats.items():
        result[vkey] = np.vstack(feat_list)

    print(f"  {sum(len(v) for v in result.values()):,} embeddings across {len(result)} videos")
    return result


def _add_video_records(records: list, vdir: Path, out_dir: Path):
    for fp in sorted(vdir.iterdir()):
        if fp.suffix.lower() in IMAGE_SUFFIXES:
            records.append({
                "frame_path": str(fp),
                "output_path": str(out_dir / fp.with_suffix(".npy").name),
            })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features with any registered model")
    parser.add_argument("--model", type=str, default="dinov2-base",
                        help="Model name (use --list to see all)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--frames-dir", type=Path,
                        default=_find_existing(_FRAMES_CANDIDATES, _FRAMES_CANDIDATES[0]))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir. Default: <embeddings_root>_<model_name>/")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Only process these category folders")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Recompute existing embeddings")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in list_extractors():
            print(f"  {name}")
        return

    if args.output_dir is None:
        emb_root = _find_existing(_EMBEDDINGS_CANDIDATES, _EMBEDDINGS_CANDIDATES[0])
        args.output_dir = emb_root.parent / f"{emb_root.name}_{args.model}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Frames: {args.frames_dir}")
    print(f"Output: {args.output_dir}")

    extractor = get_extractor(args.model)
    extractor.load_model(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_all(
        extractor,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
