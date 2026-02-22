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
import shutil
import os


def get_feature_paths(directory: Path, exclude_folders: list[str] | None = None) -> list[Path]:
    """Get all .npy feature files from a directory (recursively)."""
    all_paths = sorted(directory.rglob("*.npy"))

    if exclude_folders:
        all_paths = [p for p in all_paths if not any(ex in p.parts for ex in exclude_folders)]

    return all_paths


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
    relative = feature_path.relative_to(features_root)
    stem_path = images_root / relative.with_suffix("")

    for ext in image_extensions:
        candidate = stem_path.with_suffix(ext)
        if candidate.exists():
            return candidate
        candidate = stem_path.with_suffix(ext.upper())
        if candidate.exists():
            return candidate

    return stem_path.with_suffix(".jpg")


def find_similar_images(
        reference_features: np.ndarray,
        target_features: np.ndarray,
        target_paths: list[Path],
        threshold: float = 0.3,
) -> tuple[list[dict], np.ndarray]:
    """Find target images that are similar to any reference image."""
    print("Computing cosine distances...")
    print(target_features.shape, reference_features.shape)
    distances = cosine_distances(target_features, reference_features)
    min_distances = distances.min(axis=1)

    similar_mask = min_distances < threshold
    similar_indices = np.where(similar_mask)[0]

    results = []
    for idx in similar_indices:
        results.append({
            "feature_path": target_paths[idx],
            "min_distance": min_distances[idx],
            "closest_ref_idx": distances[idx].argmin(),
        })

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

        plt.suptitle(f"Image {i + 1}/{n_display} - Would be filtered out", fontsize=12)
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
        f"Would filter: {n_filtered}/{len(min_distances)} ({100 * n_filtered / len(min_distances):.1f}%)",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def copy_filtered_images(
        similar_results: list,
        features_root: Path,
        images_root: Path,
        output_dir: Path,
):
    """Copy filtered images to output directory (flat, no subfolders)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(similar_results)} images to: {output_dir}")

    copied = 0
    for result in tqdm(similar_results, desc="Copying images"):
        img_path = feature_path_to_image_path(result["feature_path"], features_root, images_root)

        if img_path and img_path.exists():
            # Create unique flat filename: parentfolder_filename_dist.ext
            parent_name = img_path.parent.name
            new_name = f"{parent_name}_{img_path.stem}_dist{result['min_distance']:.4f}{img_path.suffix}"
            dest_path = output_dir / new_name

            # Handle duplicates
            counter = 1
            while dest_path.exists():
                new_name = f"{parent_name}_{img_path.stem}_dist{result['min_distance']:.4f}_{counter}{img_path.suffix}"
                dest_path = output_dir / new_name
                counter += 1

            shutil.copy(img_path, dest_path)
            copied += 1
        else:
            print(f"  Warning: Image not found: {img_path}")

    print(f"Copied {copied}/{len(similar_results)} images to: {output_dir}")


def delete_filtered_images(
        similar_results: list,
        features_root: Path,
        images_root: Path,
):
    """Delete filtered images and their corresponding feature files."""
    print(f"\nDeleting {len(similar_results)} images...")

    deleted_images = 0
    deleted_features = 0

    for result in tqdm(similar_results, desc="Deleting"):
        # Delete image
        img_path = feature_path_to_image_path(result["feature_path"], features_root, images_root)
        if img_path and img_path.exists():
            os.remove(img_path)
            deleted_images += 1

        # Delete feature file
        feature_path = result["feature_path"]
        if feature_path.exists():
            os.remove(feature_path)
            deleted_features += 1

    print(f"Deleted {deleted_images} images and {deleted_features} feature files")


def main():
    # ===== CONFIGURATION =====

    # Root directories
    features_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features")
    images_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames")

    # Reference features (bad images to filter out)
    reference_features_dir = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features/reference images")

    # Target = ALL features (excluding reference folder)
    target_features_dir = features_root

    # Output folder for filtered images (flat, no subfolders)
    output_dir = Path(r"/media/HDD1/moritz/foundential/would be deleted")

    # Folders to exclude from target
    exclude_folders = ["reference images", "reference for filtering"]

    # Cosine distance threshold (0 = identical, 2 = opposite)
    threshold = 0.36

    # DELETE MODE: Set to True to delete instead of copy
    delete_filtered = True

    # =========================

    print("=" * 60)
    print("SURGICAL FRAME FILTERING (using pre-extracted features)")
    print("=" * 60)

    # Load reference features
    print("\n[1/4] Loading REFERENCE features...")
    ref_feature_paths = get_feature_paths(reference_features_dir)
    print(f"  Found {len(ref_feature_paths)} reference feature files")

    if not ref_feature_paths:
        print("ERROR: No reference features found!")
        return

    ref_features, ref_feature_paths = load_features(ref_feature_paths)
    print(f"  Reference features shape: {ref_features.shape}")

    # Load target features (all, excluding reference folders)
    print("\n[2/4] Loading TARGET features (all except reference)...")
    target_feature_paths = get_feature_paths(target_features_dir, exclude_folders=exclude_folders)
    print(f"  Found {len(target_feature_paths)} target feature files")

    if not target_feature_paths:
        print("ERROR: No target features found!")
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

    # Handle filtered images
    if similar_results:
        # COPY MODE
        user_input = input(f"\nCopy {len(similar_results)} filtered images to '{output_dir}'? (y/n): ").strip().lower()
        if user_input == "y":
            copy_filtered_images(
                similar_results,
                features_root,
                images_root,
                output_dir,
            )
        if delete_filtered:
            # DELETE MODE
            user_input = input(f"\n⚠️  PERMANENTLY DELETE {len(similar_results)} images and features? (yes/no): ").strip().lower()
            if user_input == "yes":
                delete_filtered_images(
                    similar_results,
                    features_root,
                    images_root,
                )


        # Display similar images
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