#!/usr/bin/env python3
"""
Extract frames at 1fps from videos, grouped by subdirectory.
All videos in the same folder are treated as one continuous sequence.
Frames are resized to 224x224x3.

Requirements: pip install opencv-python
"""

import argparse
import re
import traceback
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import uuid


def sanitize_folder_name(name: str) -> str:
    """
    Remove emojis and non-ASCII characters from folder name.
    Keep only alphanumeric, spaces, underscores, hyphens, and dots.
    """
    # Remove emojis and non-ASCII characters
    sanitized = name.encode('ascii', 'ignore').decode('ascii')

    # Remove any remaining problematic characters, keep alphanumeric, space, underscore, hyphen, dot
    sanitized = re.sub(r'[^\w\s\-.]', '', sanitized)

    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # If empty after sanitization, use a placeholder
    if not sanitized:
        sanitized = f"folder_{uuid.uuid4().hex[:6]}"

    return sanitized


def has_non_ascii(name: str) -> bool:
    """Check if a string contains non-ASCII characters (emojis, special chars)."""
    try:
        name.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def find_videos(root_dir: Path, extensions: list[str] = None) -> tuple[dict[str, list[Path]], set[str], dict[str, str]]:
    """
    Find all video files and group them by their immediate parent directory.
    Handles duplicate folder names by using grandparent or adding unique suffix.
    Returns: (grouped_videos, duplicated_names, group_to_original_name_mapping)
    """
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    # First pass: group videos by their parent directory path (full path to avoid confusion)
    path_grouped = defaultdict(list)

    for ext in extensions:
        for video in root_dir.rglob(f"*{ext}"):
            # Skip hidden files/directories
            if any(part.startswith(".") for part in video.parts):
                continue

            parent_path = video.parent
            path_grouped[parent_path].append(video)

    # Sort videos within each group for consistent ordering
    for key in path_grouped:
        path_grouped[key] = sorted(path_grouped[key])

    # Second pass: create unique names for each group
    # Check which parent folder names are duplicated (after sanitization)
    sanitized_names = [sanitize_folder_name(p.name) for p in path_grouped.keys()]
    name_counts = defaultdict(int)
    for name in sanitized_names:
        name_counts[name] += 1

    duplicated_names = {name for name, count in name_counts.items() if count > 1}

    # Build final grouped dict with unique names
    grouped = {}
    used_names = set()
    group_to_original = {}  # Maps final group name -> original parent name

    for parent_path, videos in path_grouped.items():
        parent_name = sanitize_folder_name(parent_path.name)
        original_name = parent_path.name

        if parent_name not in duplicated_names:
            # No conflict, use as-is
            final_name = parent_name
        else:
            # Try using grandparent_parent format
            grandparent_name = parent_path.parent.name if parent_path.parent != parent_path else ""
            grandparent_name = sanitize_folder_name(grandparent_name) if grandparent_name else ""

            if grandparent_name:
                candidate = f"{grandparent_name}_{parent_name}"
            else:
                candidate = parent_name

            # Check if this candidate is unique
            if candidate not in used_names:
                final_name = candidate
            else:
                # Still a conflict, add random suffix
                short_uuid = uuid.uuid4().hex[:6]
                final_name = f"{parent_name}_{short_uuid}"

        used_names.add(final_name)
        grouped[final_name] = videos
        group_to_original[final_name] = original_name

    return grouped, duplicated_names, group_to_original


def find_empty_groups(
        grouped_videos: dict[str, list[Path]],
        output_dir: Path,
        format: str = "jpg",
        duplicated_names: set[str] = None,
        group_to_original: dict[str, str] = None,
        exclude_duplicates: bool = False
) -> dict[str, list[Path]]:
    """
    Find groups where output folder is empty or missing but input has videos.
    Also finds groups where original folder had emojis/non-ASCII (needs re-extraction with sanitized name).
    Returns filtered dict of groups that need (re)processing.
    """
    empty_groups = {}

    for group_name, videos in grouped_videos.items():
        # Skip duplicates if requested
        if exclude_duplicates and duplicated_names and group_to_original:
            original_name = group_to_original.get(group_name, "")
            sanitized_original = sanitize_folder_name(original_name)
            if sanitized_original in duplicated_names:
                continue

        original_name = group_to_original.get(group_name, group_name) if group_to_original else group_name
        group_output_dir = output_dir / group_name

        needs_processing = False
        reason = ""

        # Check if original name had emojis/non-ASCII characters
        if has_non_ascii(original_name):
            # Check if old emoji folder exists in output
            old_emoji_dir = output_dir / original_name
            if old_emoji_dir.exists():
                needs_processing = True
                reason = "emoji folder exists, needs re-extraction"
            elif not group_output_dir.exists():
                needs_processing = True
                reason = "sanitized folder missing"
            else:
                frame_count = len(list(group_output_dir.glob(f"*.{format}")))
                if frame_count == 0:
                    needs_processing = True
                    reason = "sanitized folder empty"
        else:
            # Normal check: output dir doesn't exist or is empty
            if not group_output_dir.exists():
                needs_processing = True
                reason = "folder missing"
            else:
                frame_count = len(list(group_output_dir.glob(f"*.{format}")))
                if frame_count == 0:
                    needs_processing = True
                    reason = "folder empty"

        if needs_processing and len(videos) > 0:
            empty_groups[group_name] = (videos, reason)

    return empty_groups


def filter_by_folder_name(
        grouped_videos: dict[str, list[Path]],
        group_to_original: dict[str, str],
        folder_filter: str
) -> dict[str, list[Path]]:
    """
    Filter groups by folder name (matches sanitized name, original name, or partial match).
    """
    folder_filter_lower = folder_filter.lower()
    filtered = {}

    for group_name, videos in grouped_videos.items():
        original_name = group_to_original.get(group_name, group_name)

        # Check exact match (sanitized or original)
        if group_name.lower() == folder_filter_lower or original_name.lower() == folder_filter_lower:
            filtered[group_name] = videos
        # Check partial match (substring in sanitized or original)
        elif folder_filter_lower in group_name.lower() or folder_filter_lower in original_name.lower():
            filtered[group_name] = videos

    return filtered


def extract_frames_from_video(
        video_path: Path,
        output_dir: Path,
        start_frame_idx: int,
        size: tuple[int, int] = (224, 224),
        format: str = "jpg",
        quality: int = 95
) -> tuple[int, str | None, str | None]:
    """
    Extract 1 frame per second from a video, resized to specified size.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        start_frame_idx: Starting index for frame numbering (for continuity)
        size: Output frame size (width, height)
        format: Output image format
        quality: JPEG quality (1-100)

    Returns: (number of frames extracted, error message or None, full traceback or None)
    """
    cap = None
    try:
        cap = cv2.VideoCapture("\\\\?\\" + str(video_path), cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return 0, f"Could not open video file", f"VideoCapture failed to open: {video_path}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            return 0, f"Invalid framerate ({fps})", f"Video has invalid FPS: {fps}, total_frames: {total_frames}, path: {video_path}"

        duration_sec = total_frames / fps
        frames_extracted = 0
        second = 0

        while second < duration_sec:
            # Seek to timestamp (milliseconds)
            timestamp_ms = second * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            ret, frame = cap.read()
            if not ret:
                break

            # Resize to target size
            frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

            # Save frame with continuous numbering
            frame_idx = start_frame_idx + frames_extracted
            frame_filename = f"frame_{frame_idx:06d}.{format}"
            frame_path = output_dir / frame_filename

            if format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == "png":
                cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(str(frame_path), frame_resized)

            frames_extracted += 1
            second += 1

        return frames_extracted, None, None

    except Exception as e:
        tb = traceback.format_exc()
        return 0, f"{type(e).__name__}: {str(e)}", tb

    finally:
        if cap is not None:
            cap.release()


def process_video_group(
        group_name: str,
        videos: list[Path],
        output_dir: Path,
        size: tuple[int, int] = (224, 224),
        format: str = "jpg",
        quality: int = 95,
        verbose: bool = False
) -> tuple[str, int, int, int, list[str]]:
    """
    Process all videos in a group as one continuous sequence.

    Returns: (group_name, total_frames, successful_videos, failed_videos, log_messages)
    """
    log_messages = []

    try:
        group_output_dir = output_dir / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        tb = traceback.format_exc()
        log_messages.append(f"    ❌ Failed to create output directory: {group_output_dir}")
        log_messages.append(f"    Error: {type(e).__name__}: {str(e)}")
        log_messages.append(f"    Traceback:\n{tb}")
        return group_name, 0, 0, len(videos), log_messages

    total_frames = 0
    successful = 0
    failed = 0

    for video in videos:
        try:
            frames, error, tb = extract_frames_from_video(
                video_path=video,
                output_dir=group_output_dir,
                start_frame_idx=total_frames,
                size=size,
                format=format,
                quality=quality
            )

            if error:
                failed += 1
                log_messages.append(f"    ❌ {video.name}: {error}")
                if tb:
                    log_messages.append(f"       Full path: {video}")
                    log_messages.append(f"       Traceback:\n{tb}")
            else:
                successful += 1
                total_frames += frames
                if verbose:
                    log_messages.append(f"    ✅ {video.name}: {frames} frames (total: {total_frames})")

        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            log_messages.append(f"    ❌ {video.name}: Unexpected error")
            log_messages.append(f"       Full path: {video}")
            log_messages.append(f"       Error: {type(e).__name__}: {str(e)}")
            log_messages.append(f"       Traceback:\n{tb}")

    return group_name, total_frames, successful, failed, log_messages


def append_failed_folder(failed_log_path: Path, group_name: str, reason: str):
    """Append a failed folder to the log file."""
    with open(failed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{group_name}\t{reason}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 224x224 frames from videos grouped by subdirectory"
    )
    parser.add_argument(
        "-input",
        type=str,
        help="Input directory containing video subdirectories",
        default=r"D:\Microvascular Decompressions\Uploaded to TouchSurgery"

    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=r"D:\Extracted Frames\MVD",
        help="Output directory (default: ./frames_224)"
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=224,
        help="Output frame size (default: 224 for 224x224)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format (default: jpg)"
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--duplicates-only",
        action="store_true",
        help="Only process groups that had duplicate folder names"
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="Only process groups where output folder is empty/missing or had emojis (excludes duplicates)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=14,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "-g", "--group",
        type=str,
        default=None,
        help="Process only a specific folder/group (partial match supported)"
    )
    parser.add_argument(
        "--failed-log",
        type=str,
        default="failed_folders.txt",
        help="Path to log file for failed folders (default: failed_folders.txt)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    size = (args.size, args.size)
    failed_log_path = Path(args.failed_log)

    if not input_dir.exists():
        print(f"Error: '{input_dir}' does not exist")
        return 1

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return 1

    # Find and group videos
    print(f"Scanning for videos in: {input_dir.resolve()}")
    try:
        grouped_videos, duplicated_names, group_to_original = find_videos(input_dir)
    except Exception as e:
        print(f"\n❌ Error scanning for videos:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return 1

    if not grouped_videos:
        print("No video files found.")
        return 0

    # Report duplicates
    if duplicated_names:
        print(f"\n⚠️  Found {len(duplicated_names)} duplicate folder name(s):")
        for dup in sorted(duplicated_names):
            print(f"    - '{dup}'")
        print("    (Using grandparent folder or UUID suffix to disambiguate)\n")

    # Filter to specific group if requested
    if args.group:
        filtered = filter_by_folder_name(grouped_videos, group_to_original, args.group)

        if not filtered:
            print(f"\n❌ No groups found matching '{args.group}'")
            print("\nAvailable groups:")
            for name in sorted(grouped_videos.keys())[:20]:
                original = group_to_original.get(name, name)
                if name != original:
                    print(f"    - '{name}' (original: '{original}')")
                else:
                    print(f"    - '{name}'")
            if len(grouped_videos) > 20:
                print(f"    ... and {len(grouped_videos) - 20} more")
            return 1

        print(f"\n🎯 Matched {len(filtered)} group(s) for '{args.group}':")
        for name in sorted(filtered.keys()):
            original = group_to_original.get(name, name)
            if name != original:
                print(f"    - '{name}' (original: '{original}')")
            else:
                print(f"    - '{name}'")
        print()

        grouped_videos = filtered

    # Filter to duplicates only if requested
    if args.duplicates_only:
        # Keep only groups whose original parent name was duplicated
        filtered = {}
        for group_name, videos in grouped_videos.items():
            original_parent = group_to_original.get(group_name, "")
            sanitized_original = sanitize_folder_name(original_parent)
            if sanitized_original in duplicated_names:
                filtered[group_name] = videos
        grouped_videos = filtered

        if not grouped_videos:
            print("No duplicate groups to process.")
            return 0

        print(f"Processing only duplicate groups ({len(grouped_videos)} groups)")

    # Filter to empty/missing output folders if requested (excludes duplicates)
    if args.retry_empty:
        try:
            empty_groups = find_empty_groups(
                grouped_videos,
                output_dir,
                args.format,
                duplicated_names=duplicated_names,
                group_to_original=group_to_original,
                exclude_duplicates=True
            )
        except Exception as e:
            print(f"\n❌ Error finding empty groups:")
            print(f"   {type(e).__name__}: {str(e)}")
            print(f"\nTraceback:\n{traceback.format_exc()}")
            return 1

        if not empty_groups:
            print("No empty/missing/emoji output folders found (excluding duplicates). All groups already have frames.")
            return 0

        print(f"\n🔄 Found {len(empty_groups)} group(s) needing processing (excluding duplicates):")
        for group_name in sorted(empty_groups.keys()):
            videos, reason = empty_groups[group_name]
            print(f"    - '{group_name}' ({len(videos)} videos) [{reason}]")
        print()

        # Convert back to simple dict (remove reason)
        grouped_videos = {k: v[0] for k, v in empty_groups.items()}

    total_videos = sum(len(v) for v in grouped_videos.values())
    print(f"Found {total_videos} video(s) in {len(grouped_videos)} group(s)")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Frame size: {args.size}x{args.size}")
    print(f"Format: {args.format.upper()}")
    print(f"Workers: {args.workers}")
    print(f"Failed log: {failed_log_path.resolve()}")
    print("-" * 60)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"\n❌ Error creating output directory:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return 1

    # Process each group in parallel using multiprocessing
    grand_total_frames = 0
    grand_total_successful = 0
    grand_total_failed = 0

    sorted_groups = sorted(grouped_videos.items())
    total_groups = len(sorted_groups)

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_group = {
                executor.submit(
                    process_video_group,
                    group_name,
                    videos,
                    output_dir,
                    size,
                    args.format,
                    args.quality,
                    args.verbose
                ): (idx, group_name, len(videos))
                for idx, (group_name, videos) in enumerate(sorted_groups, 1)
            }

            # Process results as they complete
            for future in as_completed(future_to_group):
                idx, group_name, video_count = future_to_group[future]

                try:
                    result_group_name, frames, successful, failed, log_messages = future.result()

                    grand_total_frames += frames
                    grand_total_successful += successful
                    grand_total_failed += failed

                    status = "✅" if failed == 0 else "⚠️" if successful > 0 else "❌"
                    print(
                        f"[{idx}/{total_groups}] {status} 📁 {result_group_name} ({video_count} videos) → {frames} frames ({successful} ok, {failed} failed)")
                    for msg in log_messages:
                        print(msg)

                    # Log failed folders
                    if failed > 0:
                        append_failed_folder(failed_log_path, result_group_name, f"{failed} videos failed")

                except Exception as e:
                    grand_total_failed += video_count
                    tb = traceback.format_exc()
                    print(f"[{idx}/{total_groups}] ❌ 📁 {group_name} - PROCESS ERROR")
                    print(f"    Error: {type(e).__name__}: {str(e)}")
                    print(f"    Traceback:\n{tb}")

                    # Log failed folders
                    append_failed_folder(failed_log_path, group_name, f"PROCESS ERROR: {type(e).__name__}: {str(e)}")

    except Exception as e:
        print(f"\n❌ Error during parallel processing:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total groups:     {len(grouped_videos)}")
    print(f"Total videos:     {grand_total_successful + grand_total_failed}")
    print(f"  Successful:     {grand_total_successful}")
    print(f"  Failed:         {grand_total_failed}")
    print(f"Total frames:     {grand_total_frames}")
    print(f"Output:           {output_dir.resolve()}")
    if grand_total_failed > 0:
        print(f"Failed log:       {failed_log_path.resolve()}")

    # Show output structure
    print("\n" + "-" * 60)
    print("OUTPUT STRUCTURE:")
    print("-" * 60)
    for group_name in sorted(grouped_videos.keys()):
        group_dir = output_dir / group_name
        if group_dir.exists():
            frame_count = len(list(group_dir.glob(f"*.{args.format}")))
            print(f"  📁 {group_name}/ ({frame_count} frames)")

    return 0 if grand_total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())