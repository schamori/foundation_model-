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
    """Load all features for a single video in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    def _load(path):
        try:
            return np.load(path)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        features = list(ex.map(_load, feature_paths))

    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if not valid_indices:
        return None

    n_failed = len(features) - len(valid_indices)
    if n_failed > 0:
        print(f"  Warning: {n_failed}/{len(features)} feature files failed to load, filling with nearest")
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
        images_root: Path | None = None,
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

    if images_root is not None:
        allowed = {d.name for d in images_root.rglob("*") if d.is_dir()}
        videos = {name: paths for name, paths in videos.items() if name in allowed}
        print(f"Restricting to {len(videos)} videos found in {images_root}")

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
        show_viewer: bool = True,
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

    import json as _json
    import urllib.parse
    from http.server import HTTPServer, BaseHTTPRequestHandler

    # Sort by score ASCENDING (lowest first, starting from threshold)
    changes = sorted(changes, key=lambda x: x.change_score, reverse=False)

    print(f"\nBuilding viewer for {len(changes)} temporal changes...")

    saved_count = 0
    ref_feature_paths = []
    pairs = []

    for change in changes:
        pos = change.position
        paths = change.frame_paths

        frame_before_idx = pos - 1
        frame_after_idx = pos

        if frame_before_idx < 0 or frame_after_idx >= len(paths):
            continue

        try:
            emb_before = np.load(paths[frame_before_idx])
            emb_after = np.load(paths[frame_after_idx])
        except Exception:
            continue

        z_before = (emb_before - global_mean) / (global_std + 1e-8)
        z_after  = (emb_after  - global_mean) / (global_std + 1e-8)
        dev_before = float(np.linalg.norm(z_before))
        dev_after  = float(np.linalg.norm(z_after))

        if dev_before > dev_after:
            chosen_idx, chosen_label, chosen_path = frame_before_idx, "BEFORE", paths[frame_before_idx]
        else:
            chosen_idx, chosen_label, chosen_path = frame_after_idx, "AFTER", paths[frame_after_idx]

        img_before = feature_path_to_image_path(paths[frame_before_idx], features_root, images_root)
        img_after  = feature_path_to_image_path(paths[frame_after_idx],  features_root, images_root)
        chosen_img_path = feature_path_to_image_path(chosen_path, features_root, images_root)

        ref_feature_paths.append(chosen_path)

        if save_reference_images and reference_images_dir and chosen_img_path and chosen_img_path.exists():
            save_name = f"{change.video_name}_frame{chosen_idx}_score{change.change_score:.4f}{chosen_img_path.suffix}"
            shutil.copy(chosen_img_path, reference_images_dir / save_name)
            saved_count += 1

        pairs.append({
            "before": str(img_before) if img_before and img_before.exists() else "",
            "after":  str(img_after)  if img_after  and img_after.exists()  else "",
            "score":  round(change.change_score, 4),
            "video":  change.video_name,
            "chosen": chosen_label,
            "dev_before": round(dev_before, 2),
            "dev_after":  round(dev_after, 2),
        })

    if save_reference_images:
        print(f"Saved {saved_count} reference images to: {reference_images_dir}")

    pairs_json = _json.dumps(pairs)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Temporal changes</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column}}
  #bar{{padding:10px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}}
  #counter{{font-weight:bold;font-size:1.1em}}
  #score{{color:#f90;font-size:.95em}}
  #video{{color:#666;font-size:.8em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:40%}}
  #hint{{font-size:.8em;color:#555;margin-left:auto}}
  #imgs{{flex:1;display:flex;gap:8px;padding:8px;min-height:0}}
  .slot{{flex:1;display:flex;flex-direction:column;min-height:0}}
  .lbl{{font-size:.75em;color:#888;padding:2px 0 4px}}
  .lbl span{{color:#4f4;font-weight:bold}}
  .slot img{{flex:1;object-fit:contain;width:100%;min-height:0;border-radius:4px;background:#222}}
  .chosen img{{outline:3px solid #4f4;border-radius:4px}}
  .missing{{flex:1;display:flex;align-items:center;justify-content:center;color:#555;font-size:.9em;background:#1a1a1a;border-radius:4px}}
</style></head><body>
<div id="bar">
  <span id="counter"></span>
  <span id="score"></span>
  <span id="video"></span>
  <span id="hint">a / ← &nbsp;&nbsp; d / →</span>
</div>
<div id="imgs">
  <div class="slot" id="slotB"><div class="lbl" id="lblB">BEFORE</div><img id="imgB"></div>
  <div class="slot" id="slotA"><div class="lbl" id="lblA">AFTER</div><img id="imgA"></div>
</div>
<script>
var pairs={pairs_json};
var i=0;
function show(){{
  var p=pairs[i];
  document.getElementById('counter').textContent=(i+1)+'/'+pairs.length;
  document.getElementById('score').textContent='score='+p.score;
  document.getElementById('video').textContent=p.video;
  function setSlotImg(imgId, path, idx, suffix){{
    var el=document.getElementById(imgId);
    var slot=el.parentElement;
    var old=slot.querySelector('.missing');if(old)old.remove();
    if(path){{el.style.display='';el.src='/img?p='+encodeURIComponent(path)+'&i='+idx+suffix;}}
    else{{el.style.display='none';var d=document.createElement('div');d.className='missing';d.textContent='not found: '+path;slot.appendChild(d);}}
  }}
  setSlotImg('imgB',p.before,i,'b');
  setSlotImg('imgA',p.after,i,'a');
  var chB=p.chosen==='BEFORE';
  document.getElementById('slotB').className='slot'+(chB?' chosen':'');
  document.getElementById('slotA').className='slot'+(!chB?' chosen':'');
  document.getElementById('lblB').innerHTML='BEFORE &mdash; dev='+p.dev_before+(chB?' <span>✓ reference</span>':'');
  document.getElementById('lblA').innerHTML='AFTER &mdash; dev='+p.dev_after+(!chB?' <span>✓ reference</span>':'');
}}
document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='ArrowRight'){{if(i<pairs.length-1){{i++;show();}}}}
  else if(e.key==='a'||e.key==='ArrowLeft'){{if(i>0){{i--;show();}}}}
}});
show();
</script>
</body></html>"""

    if show_viewer:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode())
                elif self.path.startswith("/img?p="):
                    p = Path(urllib.parse.unquote(self.path[7:].split("&")[0]))
                    if p.exists():
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.end_headers()
                        self.wfile.write(p.read_bytes())
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)

        server = HTTPServer(("0.0.0.0", 8767), Handler)
        print(f"\nViewer running at  http://localhost:8767")
        print(f"SSH tunnel:        ssh -L 8767:localhost:8767 user@host")
        print("Press Ctrl+C to continue.\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

    return ref_feature_paths


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