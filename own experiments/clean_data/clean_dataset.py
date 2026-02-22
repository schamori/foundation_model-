"""
Filter surgical frames based on similarity to reference 'bad' images.
Uses PRE-EXTRACTED ConvNext features (.npy files) for fast comparison.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


def get_feature_paths(directory: Path) -> list[Path]:
    """Get all .npy feature files from a directory (recursively)."""
    return sorted(directory.rglob("*.npy"))


def load_features(feature_paths: list[Path], show_progress: bool = True) -> tuple[np.ndarray, list[Path]]:
    """Load features from .npy files."""
    features = []
    valid_paths = []

    iterator = tqdm(feature_paths, desc="Loading features") if show_progress else feature_paths

    for path in iterator:
        try:
            feat = np.load(path)
            features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if features:
        features = np.vstack(features)
    else:
        features = np.array([])

    return features, valid_paths


def feature_path_to_image_path(
    feature_path: Path,
    features_root: Path,
    images_root: Path,
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"),
) -> Path | None:
    """
    Convert a feature .npy path back to the original image path.
    Tries multiple extensions since we don't know the original.
    """
    # Get relative path from features root
    relative = feature_path.relative_to(features_root)

    # Remove .npy and try different image extensions
    stem_path = images_root / relative.with_suffix("")

    for ext in image_extensions:
        candidate = stem_path.with_suffix(ext)
        if candidate.exists():
            return candidate
        # Also try uppercase
        candidate = stem_path.with_suffix(ext.upper())
        if candidate.exists():
            return candidate

    # If nothing found, return the path with .jpg as fallback
    return stem_path.with_suffix(".jpg")


def find_similar_images(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    target_paths: list[Path],
    threshold: float = 0.3,
) -> tuple[list[dict], np.ndarray]:
    """Find target images that are similar to any reference image."""
    print("Computing cosine distances...")

    # Compute pairwise cosine distances
    distances = cosine_distances(target_features, reference_features)

    # Get minimum distance to any reference image for each target
    min_distances = distances.min(axis=1)

    # Find images below threshold
    similar_mask = min_distances < threshold
    similar_indices = np.where(similar_mask)[0]

    results = []
    for idx in similar_indices:
        results.append({
            "feature_path": target_paths[idx],
            "min_distance": min_distances[idx],
            "closest_ref_idx": distances[idx].argmin(),
        })

    # Sort by distance (most similar first)
    results.sort(key=lambda x: x["min_distance"])

    return results, min_distances


def display_similar_images(
    similar_results: list,
    reference_feature_paths: list[Path],
    features_root: Path,
    images_root: Path,
    ref_features_root: Path,
    ref_images_root: Path,
    max_display: int = 10000,
):
    """Display images that are similar to reference set."""
    if not similar_results:
        print("No similar images found below threshold!")
        return

    print(f"\nFound {len(similar_results)} images below threshold. Displaying up to {max_display}...")

    n_display = min(len(similar_results), max_display)

    for i, result in enumerate(similar_results[:n_display]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Target image (candidate for filtering)
        target_img_path = feature_path_to_image_path(
            result["feature_path"], features_root, images_root
        )

        try:
            target_img = Image.open(target_img_path)
            axes[0].imshow(target_img)
            axes[0].set_title(
                f"Target (dist={result['min_distance']:.4f})\n{target_img_path.name}",
                fontsize=10
            )
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Could not load:\n{target_img_path}\n{e}",
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("Target (image not found)")
        axes[0].axis("off")

        # Closest reference image
        ref_feature_path = reference_feature_paths[result["closest_ref_idx"]]
        ref_img_path = feature_path_to_image_path(
            ref_feature_path, ref_features_root, ref_images_root
        )

        try:
            ref_img = Image.open(ref_img_path)
            axes[1].imshow(ref_img)
            axes[1].set_title(f"Closest Reference\n{ref_img_path.name}", fontsize=10)
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Could not load:\n{ref_img_path}\n{e}",
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("Reference (image not found)")
        axes[1].axis("off")

        plt.suptitle(f"Image {i+1}/{n_display} - Would be filtered out", fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_distance_histogram(min_distances: np.ndarray, threshold: float):
    """Plot histogram of minimum distances to help choose threshold."""
    plt.figure(figsize=(10, 6))
    plt.hist(min_distances, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(x=threshold, color="r", linestyle="--", linewidth=2, label=f"Threshold = {threshold}")
    plt.xlabel("Minimum Cosine Distance to Reference Set")
    plt.ylabel("Count")
    plt.title("Distribution of Distances (lower = more similar to 'bad' reference images)")
    plt.legend()

    n_filtered = (min_distances < threshold).sum()
    plt.text(
        0.95, 0.95,
        f"Would filter: {n_filtered}/{len(min_distances)} ({100*n_filtered/len(min_distances):.1f}%)",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def save_filter_results(
    similar_results: list,
    all_target_paths: list[Path],
    min_distances: np.ndarray,
    features_root: Path,
    images_root: Path,
    output_path: Path,
    threshold: float,
):
    """Save filtering results to a text file."""
    with open(output_path, "w") as f:
        f.write(f"Filter Results - Threshold: {threshold}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Total images: {len(all_target_paths)}\n")
        f.write(f"To filter out: {len(similar_results)}\n")
        f.write(f"To keep: {len(all_target_paths) - len(similar_results)}\n\n")

        f.write("Images to FILTER OUT:\n")
        f.write("-" * 40 + "\n")
        for r in similar_results:
            img_path = feature_path_to_image_path(r["feature_path"], features_root, images_root)
            f.write(f"{img_path} (dist={r['min_distance']:.4f})\n")

        f.write("\n\nImages to KEEP:\n")
        f.write("-" * 40 + "\n")
        filtered_set = {r["feature_path"] for r in similar_results}
        for path, dist in zip(all_target_paths, min_distances):
            if path not in filtered_set:
                img_path = feature_path_to_image_path(path, features_root, images_root)
                f.write(f"{img_path} (dist={dist:.4f})\n")

    print(f"Results saved to: {output_path}")


def main():
    # ===== CONFIGURATION =====

    # Paths to PRE-EXTRACTED features (.npy files)
    reference_features_dir = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features/reference images")
    target_features_dir = Path(
        r"/media/HDD1/moritz/foundential/Extracted Frames Features/MVD/16.2.24_HA_L_MVD_AICA_SCA_trans_and_inter_2D_video")

    # Root directories (for path conversion)
    features_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features")
    images_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames")

    # Output file for results
    output_file = Path(r"filter_results.txt")

    # Cosine distance threshold (0 = identical, 2 = opposite)
    threshold = 0.36

    # =========================

    print("=" * 60)
    print("SURGICAL FRAME FILTERING (using pre-extracted features)")
    print("=" * 60)

    # Load reference features
    print("\n[1/4] Loading REFERENCE features...")
    ref_feature_paths = get_feature_paths(reference_features_dir)
    print(f"  Found {len(ref_feature_paths)} reference feature files")

    if not ref_feature_paths:
        print("ERROR: No reference features found! Run extract_features.py first.")
        return

    ref_features, ref_feature_paths = load_features(ref_feature_paths)
    print(f"  Reference features shape: {ref_features.shape}")

    # Load target features
    print("\n[2/4] Loading TARGET features...")
    target_feature_paths = get_feature_paths(target_features_dir)
    print(f"  Found {len(target_feature_paths)} target feature files")

    if not target_feature_paths:
        print("ERROR: No target features found! Run extract_features.py first.")
        return

    target_features, target_feature_paths = load_features(target_feature_paths)
    print(f"  Target features shape: {target_features.shape}")

    # Find similar images
    print(f"\n[3/4] Finding images with distance < {threshold}...")
    similar_results, min_distances = find_similar_images(
        ref_features, target_features, target_feature_paths, threshold
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(similar_results)} images would be filtered out")
    print(f"{'=' * 60}")

    # Show histogram
    print("\n[4/4] Displaying results...")
    plot_distance_histogram(min_distances, threshold)

    # Save results
    save_filter_results(
        similar_results,
        target_feature_paths,
        min_distances,
        features_root,
        images_root,
        output_file,
        threshold,
    )

    # Display similar images
    if similar_results:
        user_input = input("\nDisplay similar images? (y/n): ").strip().lower()
        if user_input == "y":
            display_similar_images(
                similar_results,
                ref_feature_paths,
                features_root=features_root,
                images_root=images_root,
                ref_features_root=features_root,
                ref_images_root=images_root,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total target images: {len(target_feature_paths)}")
    print(f"Images to filter (distance < {threshold}): {len(similar_results)}")
    print(f"Images to keep: {len(target_feature_paths) - len(similar_results)}")


if __name__ == "__main__":
    main()