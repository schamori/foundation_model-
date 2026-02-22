"""
LemonFM-style Cross-Video Similarity Matching

This script implements the cross-video positive pair mining strategy from LemonFM.
For each frame, it finds the closest match from DIFFERENT videos and compares it
to the temporal neighbor similarity. If cross-video match is closer than
(next_frame_similarity * factor), it's considered a good semantic match.

Reference: LemonFM uses this to find semantically similar frames across videos
for better contrastive learning.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of cross-video matching for a single frame."""
    query_path: Path
    query_video: str
    next_frame_path: Path | None
    next_frame_similarity: float | None
    best_cross_video_path: Path | None
    best_cross_video: str | None
    cross_video_similarity: float | None
    threshold: float  # next_frame_similarity * factor
    is_match: bool


def get_feature_paths_by_video(features_root: Path) -> dict[str, list[Path]]:
    """
    Group feature files by their parent folder (video).
    Returns dict: video_name -> sorted list of feature paths
    """
    all_features = sorted(features_root.rglob("*.npy"))

    videos = defaultdict(list)
    for path in all_features:
        # Use parent folder name as video identifier
        video_name = path.parent.name
        videos[video_name].append(path)

    # Sort each video's frames (assuming sequential naming)
    for video_name in videos:
        videos[video_name] = sorted(videos[video_name])

    return dict(videos)


def load_all_features(videos: dict[str, list[Path]]) -> tuple[np.ndarray, list[Path], list[str]]:
    """
    Load all features into a single array for efficient similarity computation.
    Returns: (features_array, paths_list, video_labels_list)
    """
    all_features = []
    all_paths = []
    all_video_labels = []

    print("Loading all features...")
    for video_name, paths in tqdm(videos.items()):
        for path in paths:
            try:
                feat = np.load(path)
                all_features.append(feat)
                all_paths.append(path)
                all_video_labels.append(video_name)
            except Exception as e:
                print(f"Error loading {path}: {e}")

    features_array = np.vstack(all_features)
    return features_array, all_paths, all_video_labels


def find_cross_video_matches(
        features_root: Path,
        n_samples: int = 100,
        similarity_factor: float = 3.0,
        exclude_folders: list[str] | None = None,
        seed: int = 42,
) -> list[MatchResult]:
    """
    For random frames, find cross-video matches using LemonFM strategy.

    Args:
        features_root: Root directory containing feature folders
        n_samples: Number of random frames to test
        similarity_factor: Cross-video match must be closer than (next_frame_sim * factor)
        exclude_folders: Folder names to exclude (e.g., ["reference for filtering"])
        seed: Random seed for reproducibility

    Returns:
        List of MatchResult objects
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group features by video
    videos = get_feature_paths_by_video(features_root)

    # Exclude specified folders
    if exclude_folders:
        for folder in exclude_folders:
            if folder in videos:
                del videos[folder]
                print(f"Excluded folder: {folder}")

    print(f"\nFound {len(videos)} videos:")
    for name, paths in videos.items():
        print(f"  {name}: {len(paths)} frames")

    if len(videos) < 2:
        print("ERROR: Need at least 2 videos for cross-video matching!")
        return []

    # Load all features
    all_features, all_paths, all_video_labels = load_all_features(videos)
    print(f"\nTotal features loaded: {all_features.shape}")

    # Create indices for each video
    video_indices = defaultdict(list)
    for idx, video in enumerate(all_video_labels):
        video_indices[video].append(idx)

    # Sample random frames (only from videos with >1 frame for temporal comparison)
    valid_videos = [v for v, paths in videos.items() if len(paths) > 1]

    sample_indices = []
    for _ in range(n_samples):
        # Pick random video
        video = random.choice(valid_videos)
        # Pick random frame (not the last one, so we have a next frame)
        video_idx_list = video_indices[video]
        frame_idx = random.randint(0, len(video_idx_list) - 2)
        sample_indices.append(video_idx_list[frame_idx])

    # Compute similarity matrix (query frames vs all frames)
    print(f"\nComputing similarities for {len(sample_indices)} samples...")
    sample_features = all_features[sample_indices]
    similarities = cosine_similarity(sample_features, all_features)

    # Process each sample
    results = []

    for i, query_idx in enumerate(tqdm(sample_indices, desc="Finding matches")):
        query_path = all_paths[query_idx]
        query_video = all_video_labels[query_idx]

        # Find next frame similarity (temporal neighbor)
        video_idx_list = video_indices[query_video]
        local_idx = video_idx_list.index(query_idx)

        if local_idx + 1 < len(video_idx_list):
            next_frame_idx = video_idx_list[local_idx + 1]
            next_frame_sim = similarities[i, next_frame_idx]
            next_frame_path = all_paths[next_frame_idx]
        else:
            next_frame_sim = None
            next_frame_path = None

        # Find best cross-video match
        best_cross_sim = -1
        best_cross_idx = None

        for j, (sim, video) in enumerate(zip(similarities[i], all_video_labels)):
            if video != query_video and sim > best_cross_sim:
                best_cross_sim = sim
                best_cross_idx = j

        if best_cross_idx is not None:
            best_cross_path = all_paths[best_cross_idx]
            best_cross_video = all_video_labels[best_cross_idx]
        else:
            best_cross_path = None
            best_cross_video = None
            best_cross_sim = None

        # Determine if it's a match using LemonFM criterion
        # Cross-video similarity should be > (next_frame_similarity / factor)
        # Or equivalently: cross_video_sim * factor > next_frame_sim
        if next_frame_sim is not None and best_cross_sim is not None:
            threshold = next_frame_sim / similarity_factor
            is_match = best_cross_sim > threshold
        else:
            threshold = None
            is_match = False

        results.append(MatchResult(
            query_path=query_path,
            query_video=query_video,
            next_frame_path=next_frame_path,
            next_frame_similarity=next_frame_sim,
            best_cross_video_path=best_cross_path,
            best_cross_video=best_cross_video,
            cross_video_similarity=best_cross_sim,
            threshold=threshold,
            is_match=is_match,
        ))

    return results


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


def plot_statistics(results: list[MatchResult], similarity_factor: float):
    """Plot statistics about cross-video matching."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter valid results
    valid_results = [r for r in results if r.next_frame_similarity is not None and r.cross_video_similarity is not None]

    if not valid_results:
        print("No valid results to plot!")
        return

    # 1. Match rate pie chart
    n_matches = sum(1 for r in valid_results if r.is_match)
    n_no_match = len(valid_results) - n_matches

    axes[0, 0].pie(
        [n_matches, n_no_match],
        labels=[f"Match ({n_matches})", f"No Match ({n_no_match})"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#FF5722"],
        explode=[0.05, 0],
    )
    axes[0, 0].set_title(f"Cross-Video Match Rate\n(factor={similarity_factor})")

    # 2. Similarity distributions
    next_sims = [r.next_frame_similarity for r in valid_results]
    cross_sims = [r.cross_video_similarity for r in valid_results]

    axes[0, 1].hist(next_sims, bins=30, alpha=0.7, label="Next Frame", color="#2196F3")
    axes[0, 1].hist(cross_sims, bins=30, alpha=0.7, label="Cross-Video Best", color="#FF9800")
    axes[0, 1].set_xlabel("Cosine Similarity")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Similarity Distributions")
    axes[0, 1].legend()

    # 3. Scatter: next frame sim vs cross-video sim
    colors = ["#4CAF50" if r.is_match else "#FF5722" for r in valid_results]
    axes[1, 0].scatter(next_sims, cross_sims, c=colors, alpha=0.6, s=30)

    # Add threshold line
    x_line = np.linspace(min(next_sims), max(next_sims), 100)
    y_line = x_line / similarity_factor
    axes[1, 0].plot(x_line, y_line, "k--", label=f"Threshold (next_sim/{similarity_factor})")

    axes[1, 0].set_xlabel("Next Frame Similarity")
    axes[1, 0].set_ylabel("Best Cross-Video Similarity")
    axes[1, 0].set_title("Temporal vs Cross-Video Similarity")
    axes[1, 0].legend()

    # 4. Match rate by video
    video_stats = defaultdict(lambda: {"total": 0, "matches": 0})
    for r in valid_results:
        video_stats[r.query_video]["total"] += 1
        if r.is_match:
            video_stats[r.query_video]["matches"] += 1

    videos = list(video_stats.keys())
    match_rates = [video_stats[v]["matches"] / video_stats[v]["total"] * 100 for v in videos]

    # Truncate video names for display
    video_labels = [v[:20] + "..." if len(v) > 20 else v for v in videos]

    bars = axes[1, 1].barh(video_labels, match_rates, color="#9C27B0")
    axes[1, 1].set_xlabel("Match Rate (%)")
    axes[1, 1].set_title("Match Rate by Video")
    axes[1, 1].set_xlim(0, 100)

    # Add count labels
    for bar, v in zip(bars, videos):
        count = video_stats[v]["total"]
        axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"n={count}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("cross_video_statistics.png", dpi=150)
    print("Saved statistics to: cross_video_statistics.png")
    plt.show()


def display_matches(
        results: list[MatchResult],
        features_root: Path,
        images_root: Path,
        max_display: int = 20,
        show_matches_only: bool = True,
        save_images: bool = False,
        save_dir: Path | None = None,
):
    """Display or save matching frame pairs."""
    if show_matches_only:
        to_display = [r for r in results if r.is_match]
        title_prefix = "MATCH"
    else:
        to_display = results
        title_prefix = "Sample"

    if not to_display:
        print("No matches to display!")
        return

    # If saving, limit to top 10
    if save_images:
        n_display = min(len(to_display), 10)
        if save_dir is None:
            save_dir = Path("video_matching")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving top {n_display} {'matches' if show_matches_only else 'samples'} to {save_dir}...")
    else:
        n_display = min(len(to_display), max_display)
        print(f"\nDisplaying {n_display} {'matches' if show_matches_only else 'samples'}...")

    for i, result in enumerate(to_display[:n_display]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Query frame
        query_img_path = feature_path_to_image_path(result.query_path, features_root, images_root)
        try:
            query_img = Image.open(query_img_path)
            axes[0].imshow(query_img)
        except:
            axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center")
        axes[0].set_title(f"Query Frame\n{result.query_video[:30]}", fontsize=10)
        axes[0].axis("off")

        # Next frame (temporal neighbor)
        if result.next_frame_path:
            next_img_path = feature_path_to_image_path(result.next_frame_path, features_root, images_root)
            try:
                next_img = Image.open(next_img_path)
                axes[1].imshow(next_img)
            except:
                axes[1].text(0.5, 0.5, "Image not found", ha="center", va="center")
        axes[1].set_title(f"Next Frame (sim={result.next_frame_similarity:.4f})\nThreshold: {result.threshold:.4f}",
                          fontsize=10)
        axes[1].axis("off")

        # Cross-video match
        if result.best_cross_video_path:
            cross_img_path = feature_path_to_image_path(result.best_cross_video_path, features_root, images_root)
            try:
                cross_img = Image.open(cross_img_path)
                axes[2].imshow(cross_img)
            except:
                axes[2].text(0.5, 0.5, "Image not found", ha="center", va="center")

        match_status = "✓ MATCH" if result.is_match else "✗ NO MATCH"
        axes[2].set_title(
            f"Cross-Video Best (sim={result.cross_video_similarity:.4f})\n{result.best_cross_video[:30]}\n{match_status}",
            fontsize=10)
        axes[2].axis("off")

        plt.suptitle(f"{title_prefix} {i + 1}/{n_display}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_images:
            save_path = save_dir / f"match_{i + 1:02d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    if save_images:
        print(f"Saved {n_display} images to {save_dir}/")


def main():
    # ===== CONFIGURATION =====
    features_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features")
    images_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames")
    # LemonFM matching parameters
    n_samples = 200  # Number of random frames to test
    similarity_factor = 1.13 # Cross-video match threshold factor

    # Folders to exclude from matching
    exclude_folders = ["reference for filtering"]

    # Display settings
    max_display = 20
    show_matches_only = True
    save_images = True 
    # =========================

    print("=" * 60)
    print("LEMONFM-STYLE CROSS-VIDEO MATCHING")
    print("=" * 60)
    print(f"\nCriterion: cross_video_sim > next_frame_sim / {similarity_factor}")
    print("(Cross-video match must be at least 1/{similarity_factor} as similar as temporal neighbor)")

    # Find cross-video matches
    results = find_cross_video_matches(
        features_root=features_root,
        n_samples=n_samples,
        similarity_factor=similarity_factor,
        exclude_folders=exclude_folders,
    )

    if not results:
        print("No results generated!")
        return

    # Statistics
    valid_results = [r for r in results if r.is_match is not None]
    n_matches = sum(1 for r in valid_results if r.is_match)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {len(valid_results)}")
    print(f"Cross-video matches: {n_matches} ({100 * n_matches / len(valid_results):.1f}%)")
    print(f"No matches: {len(valid_results) - n_matches}")

    # Plot statistics
    plot_statistics(results, similarity_factor)

    # Display or save matches
    if save_images:
        display_matches(
            results,
            features_root,
            images_root,
            max_display=max_display,
            show_matches_only=show_matches_only,
            save_images=True,
        )
    else:
        user_input = input(f"\nDisplay {'matches' if show_matches_only else 'samples'}? (y/n): ").strip().lower()
        if user_input == "y":
            display_matches(
                results,
                features_root,
                images_root,
                max_display=max_display,
                show_matches_only=show_matches_only,
            )

    # Print some example matches
    print("\n" + "=" * 60)
    print("EXAMPLE MATCHES")
    print("=" * 60)
    matches = [r for r in results if r.is_match][:10]
    for i, m in enumerate(matches):
        print(f"\n{i + 1}. Query: {m.query_path.name} ({m.query_video[:25]})")
        print(f"   Next frame sim: {m.next_frame_similarity:.4f}")
        print(f"   Cross-video sim: {m.cross_video_similarity:.4f} ({m.best_cross_video[:25]})")
        print(f"   Threshold: {m.threshold:.4f} → MATCH ✓")


if __name__ == "__main__":
    main()