"""
Temporal Discontinuity Detection

Finds the biggest temporal changes in surgical videos by comparing
the average features of N frames before vs N frames after each position.
Useful for detecting:
- Scene cuts
- Surgical phase transitions
- Camera switches
- Significant events
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import shutil


@dataclass
class TemporalChange:
    """Represents a detected temporal discontinuity."""
    position: int  # Frame index where the change happens
    change_score: float  # Cosine distance between before/after windows
    video_name: str
    frame_paths: list[Path]  # All frame paths in this video
    window_size: int


def get_feature_paths_by_video(features_root: Path) -> dict[str, list[Path]]:
    """Group feature files by their parent folder (video)."""
    all_features = sorted(features_root.rglob("*.npy"))

    videos = defaultdict(list)
    for path in all_features:
        video_name = path.parent.name
        videos[video_name].append(path)

    # Sort each video's frames
    for video_name in videos:
        videos[video_name] = sorted(videos[video_name])

    return dict(videos)


def load_video_features(feature_paths: list[Path]) -> np.ndarray:
    """Load all features for a single video."""
    features = []
    for path in feature_paths:
        try:
            feat = np.load(path)
            features.append(feat)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            features.append(None)

    # Handle any failed loads by interpolating
    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if not valid_indices:
        return None

    # Replace None with nearest valid feature
    for i, f in enumerate(features):
        if f is None:
            nearest = min(valid_indices, key=lambda x: abs(x - i))
            features[i] = features[nearest]

    return np.vstack(features)


def compute_temporal_changes(
        features: np.ndarray,
        window_size: int = 10,
) -> np.ndarray:
    """
    Compute temporal change scores for each position.

    For position i, computes cosine distance between:
    - mean(features[i-window_size:i])  (previous window)
    - mean(features[i:i+window_size])  (next window)

    Returns array of change scores (length = n_frames - 2*window_size + 1)
    """
    n_frames = len(features)

    if n_frames < 2 * window_size:
        return np.array([])

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    features_norm = features / norms

    change_scores = []

    # Slide through valid positions
    for i in range(window_size, n_frames - window_size + 1):
        # Previous window: [i-window_size, i)
        prev_window = features_norm[i - window_size:i]
        prev_mean = prev_window.mean(axis=0)
        prev_mean = prev_mean / (np.linalg.norm(prev_mean) + 1e-8)

        # Next window: [i, i+window_size)
        next_window = features_norm[i:i + window_size]
        next_mean = next_window.mean(axis=0)
        next_mean = next_mean / (np.linalg.norm(next_mean) + 1e-8)

        # Cosine distance (1 - cosine_similarity)
        cosine_sim = np.dot(prev_mean, next_mean)
        cosine_dist = 1 - cosine_sim

        change_scores.append(cosine_dist)

    return np.array(change_scores)


def find_all_temporal_changes(
        features_root: Path,
        window_size: int = 10,
        min_gap: int = 20,
        exclude_folders: list[str] | None = None,
        folder_filter: str | None = None,
) -> tuple[list[TemporalChange], np.ndarray, np.ndarray, np.ndarray]:
    """
    Find ALL temporal discontinuities across all videos.

    Args:
        features_root: Root directory containing feature files
        window_size: Number of frames before/after to compare
        min_gap: Minimum gap between detected changes
        exclude_folders: List of folder names to exclude
        folder_filter: If provided, only process folders containing this string (case-insensitive)

    Returns:
        - List of TemporalChange objects (local maxima with min_gap)
        - Array of ALL change scores for distribution plotting
        - Global mean embedding (across all frames)
        - Global std embedding (across all frames)
    """
    videos = get_feature_paths_by_video(features_root)

    if exclude_folders:
        for folder in exclude_folders:
            if folder in videos:
                del videos[folder]
                print(f"Excluded: {folder}")

    # Filter by folder name if specified
    if folder_filter:
        folder_filter_lower = folder_filter.lower()
        filtered_videos = {
            name: paths for name, paths in videos.items()
            if folder_filter_lower in name.lower()
        }

        if not filtered_videos:
            print(f"\n❌ No folders found matching '{folder_filter}'")
            print("\nAvailable folders:")
            for name in sorted(videos.keys())[:20]:
                print(f"    - '{name}'")
            if len(videos) > 20:
                print(f"    ... and {len(videos) - 20} more")
            return [], np.array([]), np.array([]), np.array([])

        print(f"\n🎯 Matched {len(filtered_videos)} folder(s) for '{folder_filter}':")
        for name in sorted(filtered_videos.keys()):
            print(f"    - '{name}'")
        print()

        videos = filtered_videos

    print(f"\nProcessing {len(videos)} videos with window_size={window_size}")

    all_changes = []
    all_scores = []  # Store ALL scores for distribution
    all_embeddings = []  # Store ALL embeddings for global stats

    for video_name, feature_paths in tqdm(videos.items(), desc="Analyzing videos"):
        if len(feature_paths) < 2 * window_size:
            print(f"  Skipping {video_name}: only {len(feature_paths)} frames (need {2 * window_size})")
            continue

        # Load features
        features = load_video_features(feature_paths)
        if features is None:
            continue

        # Store all embeddings for global stats
        all_embeddings.append(features)

        # Compute change scores
        change_scores = compute_temporal_changes(features, window_size)

        if len(change_scores) == 0:
            continue

        # Store all scores for distribution
        all_scores.extend(change_scores.tolist())

        # Find local maxima with minimum gap
        positions = np.argsort(change_scores)[::-1]  # Descending

        selected = []
        for pos in positions:
            actual_pos = pos + window_size

            too_close = any(abs(actual_pos - s) < min_gap for s in selected)
            if not too_close:
                selected.append(actual_pos)

                all_changes.append(TemporalChange(
                    position=actual_pos,
                    change_score=change_scores[pos],
                    video_name=video_name,
                    frame_paths=feature_paths,
                    window_size=window_size,
                ))

    # Sort all changes by score
    all_changes.sort(key=lambda x: x.change_score, reverse=True)

    # Compute global embedding stats
    all_embeddings_stacked = np.vstack(all_embeddings)
    global_mean = np.mean(all_embeddings_stacked, axis=0)
    global_std = np.std(all_embeddings_stacked, axis=0)

    print(f"Computed global stats from {len(all_embeddings_stacked):,} embeddings")

    return all_changes, np.array(all_scores), global_mean, global_std


def feature_path_to_image_path(
        feature_path: Path,
        features_root: Path,
        images_root: Path,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"),
) -> Path | None:
    """Convert feature path to image path."""
    relative = feature_path.relative_to(features_root)
    stem_path = images_root / relative.with_suffix("")

    for ext in extensions:
        for e in [ext, ext.upper()]:
            candidate = stem_path.with_suffix(e)
            if candidate.exists():
                return candidate

    return stem_path.with_suffix(".jpg")


def plot_full_distribution(all_scores: np.ndarray, changes: list[TemporalChange], threshold: float | None = None):
    """Plot distribution of ALL change scores with optional threshold line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of ALL scores
    axes[0].hist(all_scores, bins=100, edgecolor="black", alpha=0.7, color="#2196F3")
    axes[0].set_xlabel("Temporal Change Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution of ALL {len(all_scores):,} Change Scores")

    if threshold is not None:
        axes[0].axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold}")
        axes[0].legend()

    # CDF plot
    sorted_scores = np.sort(all_scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cdf, color="#4CAF50", linewidth=2)
    axes[1].set_xlabel("Temporal Change Score")
    axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title("Cumulative Distribution")
    axes[1].grid(True, alpha=0.3)

    if threshold is not None:
        axes[1].axvline(x=threshold, color="red", linestyle="--", linewidth=2)
        # Show what percentile the threshold is
        percentile = (all_scores >= threshold).sum() / len(all_scores) * 100
        axes[1].text(threshold, 0.5, f"  {percentile:.1f}% above", color="red", fontsize=10)

    plt.tight_layout()
    plt.savefig("all_scores_distribution.png", dpi=150)
    print("Saved: all_scores_distribution.png")
    plt.show()

    # Print statistics
    print(f"\nScore Statistics:")
    print(f"  Min:    {all_scores.min():.4f}")
    print(f"  Max:    {all_scores.max():.4f}")
    print(f"  Mean:   {all_scores.mean():.4f}")
    print(f"  Median: {np.median(all_scores):.4f}")
    print(f"  Std:    {all_scores.std():.4f}")
    print(f"\nPercentiles:")
    for p in [90, 95, 99, 99.5, 99.9]:
        print(f"  {p}th: {np.percentile(all_scores, p):.4f}")


def plot_filtered_distribution(changes: list[TemporalChange], threshold: float):
    """Plot distribution of changes above threshold."""
    filtered = [c for c in changes if c.change_score >= threshold]

    if not filtered:
        print(f"No changes above threshold {threshold}")
        return filtered

    scores = [c.change_score for c in filtered]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of filtered scores
    axes[0].hist(scores, bins=min(50, len(scores)), edgecolor="black", alpha=0.7, color="#FF5722")
    axes[0].set_xlabel("Temporal Change Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution of {len(filtered)} Changes ≥ {threshold}")
    axes[0].axvline(x=threshold, color="red", linestyle="--", linewidth=2)

    # Per-video breakdown
    video_counts = defaultdict(int)
    video_max_score = defaultdict(float)
    for c in filtered:
        video_counts[c.video_name] += 1
        video_max_score[c.video_name] = max(video_max_score[c.video_name], c.change_score)

    videos = sorted(video_counts.keys(), key=lambda v: video_max_score[v], reverse=True)[:20]  # Top 20 videos
    counts = [video_counts[v] for v in videos]
    video_labels = [v[:25] + "..." if len(v) > 25 else v for v in videos]

    bars = axes[1].barh(video_labels, counts, color="#4CAF50")
    axes[1].set_xlabel("Number of Changes")
    axes[1].set_title(f"Changes per Video (top 20)")

    plt.tight_layout()
    plt.savefig("filtered_distribution.png", dpi=150)
    print("Saved: filtered_distribution.png")
    plt.show()

    return filtered


def display_temporal_changes(
        changes: list[TemporalChange],
        features_root: Path,
        images_root: Path,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        save_reference_images: bool = False,
        reference_images_dir: Path | None = None,
        n_context: int = 1,  # Ignored now, always shows 2 frames
):
    """Display the detected temporal changes - shows current and next frame, picks higher deviation from global mean as reference."""
    if not changes:
        print("No changes to display!")
        return

    # Create reference images directory if saving
    if save_reference_images and reference_images_dir:
        reference_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving reference images to: {reference_images_dir}")

    # Sort by score ASCENDING (lowest first, starting from threshold)
    changes = sorted(changes, key=lambda x: x.change_score, reverse=False)

    print(f"\nDisplaying {len(changes)} temporal changes (sorted by score, lowest first)...")

    saved_count = 0

    for i, change in enumerate(changes):
        pos = change.position
        paths = change.frame_paths
        ws = change.window_size

        # Always show just 2 frames: before (pos-1) and after (pos)
        frame_before_idx = pos - 1
        frame_after_idx = pos

        if frame_before_idx < 0 or frame_after_idx >= len(paths):
            print(f"Skipping change at {pos}: out of bounds")
            continue

        # Load embeddings for both frames
        try:
            emb_before = np.load(paths[frame_before_idx])
            emb_after = np.load(paths[frame_after_idx])
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            continue

        # Compute deviation from global mean (z-score magnitude)
        # Higher = more unusual / more deviation from average
        z_before = (emb_before - global_mean) / (global_std + 1e-8)
        z_after = (emb_after - global_mean) / (global_std + 1e-8)

        dev_before = np.linalg.norm(z_before)  # L2 norm of z-scores
        dev_after = np.linalg.norm(z_after)

        # Pick the one with higher deviation as reference
        if dev_before > dev_after:
            chosen_idx = frame_before_idx
            chosen_label = "BEFORE"
            chosen_path = paths[frame_before_idx]
        else:
            chosen_idx = frame_after_idx
            chosen_label = "AFTER"
            chosen_path = paths[frame_after_idx]

        # Get image paths
        img_path_before = feature_path_to_image_path(paths[frame_before_idx], features_root, images_root)
        img_path_after = feature_path_to_image_path(paths[frame_after_idx], features_root, images_root)
        chosen_img_path = feature_path_to_image_path(chosen_path, features_root, images_root)

        # Save reference image if enabled
        if save_reference_images and reference_images_dir and chosen_img_path and chosen_img_path.exists():
            # Create unique filename: video_name_frame_idx_score.ext
            save_name = f"{change.video_name}_frame{chosen_idx}_score{change.change_score:.4f}{chosen_img_path.suffix}"
            save_path = reference_images_dir / save_name
            shutil.copy(chosen_img_path, save_path)
            saved_count += 1

        # Create figure with 2 frames
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Frame before
        try:
            img_before = Image.open(img_path_before)
            axes[0].imshow(img_before)
        except:
            axes[0].text(0.5, 0.5, "Not found", ha="center", va="center")

        is_chosen_before = (chosen_idx == frame_before_idx)
        title_before = f"BEFORE (Frame {frame_before_idx})\nGlobal Dev: {dev_before:.2f}"
        if is_chosen_before:
            title_before += "\n✓ CHOSEN AS REFERENCE"
            for spine in axes[0].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(4)
        axes[0].set_title(title_before, fontsize=10, fontweight="bold" if is_chosen_before else "normal",
                         color="green" if is_chosen_before else "black")
        axes[0].axis("off")

        # Frame after
        try:
            img_after = Image.open(img_path_after)
            axes[1].imshow(img_after)
        except:
            axes[1].text(0.5, 0.5, "Not found", ha="center", va="center")

        is_chosen_after = (chosen_idx == frame_after_idx)
        title_after = f"AFTER (Frame {frame_after_idx})\nGlobal Dev: {dev_after:.2f}"
        if is_chosen_after:
            title_after += "\n✓ CHOSEN AS REFERENCE"
            for spine in axes[1].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(4)
        axes[1].set_title(title_after, fontsize=10, fontweight="bold" if is_chosen_after else "normal",
                         color="green" if is_chosen_after else "black")
        axes[1].axis("off")

        plt.suptitle(
            f"#{i + 1}/{len(changes)}: {change.video_name[:50]}\nChange Score: {change.change_score:.4f} | Reference: {chosen_label} frame",
            fontsize=11,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    if save_reference_images:
        print(f"\nSaved {saved_count} reference images to: {reference_images_dir}")


def main():
    # ===== CONFIGURATION =====
    images_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames")
    features_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features")
    reference_images_dir = Path(r"/media/HDD1/moritz/foundential/Extracted Frames/reference images")
    folder_filter = "7.6.24_MVD22_LZ_2D_video"


    # Detection parameters
    window_size = 2
    min_gap = 1

    # Display settings
    n_context = 1
    max_display = 200  # Only visualize this many changes
    save_reference_images = False  # Set to True to save chosen reference images

    # Folders to exclude
    exclude_folders = ["reference for filtering", "reference images"]
    # =========================

    print("=" * 60)
    print("TEMPORAL DISCONTINUITY DETECTION")
    print("=" * 60)

    # Find ALL temporal changes
    all_changes, all_scores, global_mean, global_std = find_all_temporal_changes(
        features_root=features_root,
        window_size=window_size,
        min_gap=min_gap,
        exclude_folders=exclude_folders,
        folder_filter=folder_filter,
    )

    if len(all_scores) == 0:
        print("No scores computed!")
        return

    print(f"\nFound {len(all_changes)} local maxima changes")
    print(f"Total score comparisons: {len(all_scores):,}")

    plot_full_distribution(all_scores, all_changes)

    # Get threshold from user
    while True:
        try:
            threshold_input = input(f"\nEnter threshold (or 'q' to quit): ").strip()
            if threshold_input.lower() == 'q':
                break

            threshold = float(threshold_input)

            # Filter and sort ascending (lowest first)
            filtered_changes = [c for c in all_changes if c.change_score >= threshold]
            filtered_changes.sort(key=lambda x: x.change_score, reverse=False)

            if not filtered_changes:
                print(f"No changes above threshold {threshold}")
                continue

            # Plot filtered distribution
            plot_filtered_distribution(all_changes, threshold)

            print(f"\n{len(filtered_changes)} changes ≥ {threshold}")
            print(f"Will display first {min(max_display, len(filtered_changes))} (lowest scores first)")

            for i, c in enumerate(filtered_changes[:20]):
                print(f"  {i + 1}. Score: {c.change_score:.4f} | {c.video_name[:40]} | Frame {c.position}")

            # Ask to visualize
            viz = input(f"\nVisualize {min(max_display, len(filtered_changes))} changes? (y/n): ").strip().lower()
            if viz == 'y':
                display_temporal_changes(
                    filtered_changes[:max_display],  # Only show max_display
                    features_root,
                    images_root,
                    global_mean,
                    global_std,
                    save_reference_images=save_reference_images,
                    reference_images_dir=reference_images_dir,
                    n_context=n_context,
                )
        except ValueError:
            print("Invalid threshold. Enter a number.")


if __name__ == "__main__":
    main()