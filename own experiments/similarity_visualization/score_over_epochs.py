import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from visualize_similarity import compute_similarity_scores


def _collect_scores(image_path, checkpoint_dir, device, epochs, n_points):
    sws_vals = []
    aug_stability = {}

    for epoch in epochs:
        ckpt = checkpoint_dir / f"dino_epoch{epoch}.pt"
        if not ckpt.exists():
            print(f"Skipping missing checkpoint: {ckpt}")
            continue

        scores = compute_similarity_scores(
            image_path=image_path,
            checkpoint_path=ckpt,
            device=device,
            n_points=n_points
        )

        sws_vals.append((epoch, scores["structure_weighted_spread"]))

        for name, val in scores["augmentation_stability"].items():
            aug_stability.setdefault(name, []).append((epoch, val))

    return sws_vals, aug_stability


def _plot_series(ax, series, title, ylabel):
    for label, vals in series.items():
        epochs = [e for e, _ in vals]
        values = [v for _, v in vals]
        ax.plot(epochs, values, marker="o", label=str(label))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot similarity scores over DINO epochs.")
    parser.add_argument(
        "--image",
        default="/media/HDD1/moritz/foundential/Extracted Frames/MVD/MVD_056_R_21576165_02-12-2022/frame_000447.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Directory containing dino_epoch*.pt files"
    )
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=22)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument(
        "--out",
        default="score_over_epochs.png",
        help="Path to save the plot PNG"
    )
    args = parser.parse_args()

    epochs = list(range(args.start_epoch, args.end_epoch + 1))
    sws_vals, aug_stability = _collect_scores(
        image_path=args.image,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        epochs=epochs,
        n_points=args.n_points
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    _plot_series(
        axes[0],
        {"structure_weighted_spread": sws_vals},
        "Structure-Weighted Spread",
        "Spread Score"
    )
    _plot_series(
        axes[1],
        aug_stability,
        "Augmentation Stability",
        "Correlation"
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
