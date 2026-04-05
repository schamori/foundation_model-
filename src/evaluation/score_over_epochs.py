"""
Track and plot evaluation metrics across training checkpoints.

Reads from cfg.eval_dir/summary.json (written by run_all_evaluations)
and generates plots.

Usage:
    python -m src.evaluation.score_over_epochs --config Full
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics_over_epochs(eval_dir: Path, output_path: Path | None = None):
    """
    Read summary.json and plot SWS + augmentation stability over epochs.
    """
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        print(f"[score_over_epochs] No summary found at {summary_path}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    if not summary:
        print("[score_over_epochs] Empty summary")
        return

    epochs = []
    sws_vals = []
    stability_series: dict[str, list[float]] = {}

    for entry in summary:
        ep = entry["epoch"]
        sim = entry.get("metrics", {}).get("similarity", {})
        if "error" in sim:
            continue
        epochs.append(ep)
        sws_vals.append(sim.get("sws", 0))
        for key, val in sim.get("augmentation_stability", {}).items():
            stability_series.setdefault(key, []).append(val)

    if not epochs:
        print("[score_over_epochs] No valid similarity metrics found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # SWS
    axes[0].plot(epochs, sws_vals, "o-", color="#2196F3")
    axes[0].set_title("Structure-Weighted Spread")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("SWS Score")
    axes[0].grid(True, alpha=0.3)

    # Augmentation stability
    for label, vals in stability_series.items():
        axes[1].plot(epochs[:len(vals)], vals, "o-", label=label)
    axes[1].set_title("Augmentation Stability")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(True, alpha=0.3)
    if stability_series:
        axes[1].legend()

    plt.tight_layout()
    if output_path is None:
        output_path = eval_dir / "metrics_over_epochs.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[score_over_epochs] Saved → {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot evaluation metrics over epochs")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Path to eval directory (e.g., output/full/eval)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plot_metrics_over_epochs(args.eval_dir, args.output)


if __name__ == "__main__":
    main()
