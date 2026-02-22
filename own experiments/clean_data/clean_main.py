"""
Complete pipeline: Find reference images → Extract features → Filter dataset
"""

from pathlib import Path
import shutil

# Import from existing modules
from find_reference_for_cleanup import (
    find_all_temporal_changes,
    display_temporal_changes,
    plot_full_distribution,
    plot_filtered_distribution)

from extract_features import load_model, extract_and_save_features

from filter_images import (
    get_feature_paths,
    load_features,
    find_similar_images,
    plot_distance_histogram,
    delete_filtered_images,
    display_similar_images,
)


def main():
    # ===== CONFIGURATION =====
    images_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames")
    features_root = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features")
    reference_images_dir = images_root / "reference images"
    reference_features_dir = features_root / "reference images"

    exclude_folders = ["reference for filtering", "reference images"]

    # Temporal detection params
    window_size = 2
    min_gap = 1
    max_display = 200

    # Filter threshold
    filter_threshold = 0.36
    # =========================

    # Load model once
    print("Loading model...")
    model, processor, device = load_model()

    loop_through = input("Loop through automatically after first run? (y/n): ").strip().lower() == 'y'
    first_run = True
    saved_threshold = None

    while True:
        # Clear reference folders for fresh run
        if reference_images_dir.exists():
            shutil.rmtree(reference_images_dir)
        if reference_features_dir.exists():
            shutil.rmtree(reference_features_dir)

        # ==================== STEP 1: Find reference images ====================
        print("\n" + "=" * 60)
        print("STEP 1: TEMPORAL DISCONTINUITY DETECTION")
        print("=" * 60)

        all_changes, all_scores, global_mean, global_std = find_all_temporal_changes(
            features_root=features_root,
            window_size=window_size,
            min_gap=min_gap,
            exclude_folders=exclude_folders,
        )

        if len(all_scores) == 0:
            print("No scores computed! Dataset might be clean.")
            break

        if first_run:
            plot_full_distribution(all_scores, all_changes)

            threshold_input = input("\nEnter threshold for reference images (or 'q' to quit): ").strip()
            if threshold_input.lower() == 'q':
                break

            try:
                saved_threshold = float(threshold_input)
            except ValueError:
                print("Invalid threshold.")
                continue

        threshold = saved_threshold

        filtered_changes = [c for c in all_changes if c.change_score >= threshold]
        filtered_changes.sort(key=lambda x: x.change_score, reverse=False)

        if not filtered_changes:
            print(f"No changes found ≥ {threshold}. Dataset is clean!")
            break

        if first_run:
            plot_filtered_distribution(all_changes, threshold)

        print(f"\n{len(filtered_changes)} changes ≥ {threshold}")

        # Auto-save reference images
        display_temporal_changes(
            filtered_changes[:max_display],
            features_root,
            images_root,
            global_mean,
            global_std,
            save_reference_images=True,
            reference_images_dir=reference_images_dir,
        )

        # ==================== STEP 2: Extract features for reference images ====================
        print("\n" + "=" * 60)
        print("STEP 2: EXTRACT FEATURES FOR REFERENCE IMAGES")
        print("=" * 60)

        extract_and_save_features(
            input_dir=reference_images_dir,
            output_dir=reference_features_dir,
            model=model,
            processor=processor,
            device=device,
            batch_size=512,
            skip_existing=True,
        )

        # ==================== STEP 3: Filter dataset ====================
        print("\n" + "=" * 60)
        print("STEP 3: FILTER DATASET")
        print("=" * 60)

        ref_feature_paths = get_feature_paths(reference_features_dir)
        ref_features, ref_feature_paths = load_features(ref_feature_paths)

        target_feature_paths = get_feature_paths(features_root, exclude_folders=exclude_folders)
        target_features, target_feature_paths = load_features(target_feature_paths)

        similar_results, min_distances = find_similar_images(
            ref_features, target_features, target_feature_paths, filter_threshold
        )

        print(f"\nFound {len(similar_results)} images to filter")

        if first_run:
            plot_distance_histogram(min_distances, filter_threshold)

        if not similar_results:
            print("No similar images found. Dataset is clean!")
            break

        if first_run:
            # Display images to be deleted
            if input(f"\nDisplay {len(similar_results)} images to be deleted? (y/n): ").strip().lower() == 'y':
                display_similar_images(
                    similar_results,
                    ref_feature_paths,
                    features_root=features_root,
                    images_root=images_root,
                    ref_features_root=features_root,
                    ref_images_root=images_root,
                    max_display=len(similar_results),
                )

            # Delete
            if input(f"\n⚠️ DELETE {len(similar_results)} images? (yes/no): ").strip().lower() == 'yes':
                delete_filtered_images(similar_results, features_root, images_root)
                print("\n🔄 Restarting pipeline to find more...\n")
                first_run = not loop_through  # If loop_through, set first_run=False
            else:
                break
        else:
            # Auto mode - just delete
            delete_filtered_images(similar_results, features_root, images_root)
            print("\n🔄 Auto-looping to find more...\n")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()