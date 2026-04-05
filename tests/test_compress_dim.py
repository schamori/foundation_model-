"""
Test: dinov2-small (384-dim native) vs dinov2-base (768-dim → PCA 384).

Compares splitting quality when dinov2-base is compressed to match dinov2-small's
native dimensionality.

Usage:
    python tests/test_compress_dim.py
    python tests/test_compress_dim.py --categories MVD
    python tests/test_compress_dim.py --n-trials 50000
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import _find_existing, _FRAMES_CANDIDATES, PROJECT_ROOT
from src.datasplitting.evaluate import run_evaluation, print_comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test dim reduction: dinov2-small vs dinov2-base@384")
    parser.add_argument("--frames-dir", type=Path,
                        default=_find_existing(_FRAMES_CANDIDATES, _FRAMES_CANDIDATES[0]))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "output" / "datasplitting"

    # Run 1: dinov2-small at native 384-dim (no compression)
    print("\n" + "#" * 70)
    print("# RUN 1: dinov2-small (native 384-dim)")
    print("#" * 70)
    results_small = run_evaluation(
        models=["dinov2-small"],
        frames_dir=args.frames_dir,
        output_root=output_root,
        categories=args.categories,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        no_save=True,
    )

    # Run 2: dinov2-base compressed to 384-dim via PCA
    print("\n" + "#" * 70)
    print("# RUN 2: dinov2-base (768-dim → PCA 384)")
    print("#" * 70)
    results_base = run_evaluation(
        models=["dinov2-base"],
        frames_dir=args.frames_dir,
        output_root=output_root,
        categories=args.categories,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        no_save=True,
        compress_dim=384,
    )

    # Merge and compare
    all_results = {}
    all_results["dinov2-small (native 384)"] = results_small["dinov2-small"]
    all_results["dinov2-small (native 384)"]["model"] = "dinov2-small (native 384)"
    all_results["dinov2-base (PCA 384)"] = results_base["dinov2-base"]
    all_results["dinov2-base (PCA 384)"]["model"] = "dinov2-base (PCA 384)"

    print_comparison(all_results)


if __name__ == "__main__":
    main()
