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
import json as _json_mod


# ── Anti-reference helpers ──────────────────────────────────────────

def load_anti_references(anti_ref_file: Path) -> tuple[list[Path], np.ndarray | None]:
    """Load anti-reference feature paths and their embeddings from disk."""
    if not anti_ref_file or not anti_ref_file.exists():
        return [], None
    with open(anti_ref_file) as f:
        data = _json_mod.load(f)
    paths = [Path(p) for p in data.get("paths", [])]
    embeddings = []
    valid = []
    for p in paths:
        if p.exists():
            try:
                embeddings.append(np.load(p).astype(np.float32))
                valid.append(p)
            except Exception:
                pass
    if not embeddings:
        return valid, None
    return valid, np.vstack(embeddings)


def save_anti_references(anti_ref_file: Path, paths: list[Path]):
    """Save anti-reference feature paths to disk (deduplicates)."""
    unique = list(dict.fromkeys(str(p) for p in paths))
    anti_ref_file.parent.mkdir(parents=True, exist_ok=True)
    with open(anti_ref_file, "w") as f:
        _json_mod.dump({"paths": unique}, f)
    print(f"[anti-ref] Saved {len(unique)} anti-reference paths to {anti_ref_file}")


def _cosine_dist_to_anti_refs(embedding: np.ndarray, anti_ref_embeddings: np.ndarray) -> float:
    """Return minimum cosine distance between embedding and any anti-reference."""
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    norms = np.linalg.norm(anti_ref_embeddings, axis=1, keepdims=True) + 1e-8
    anti_norm = anti_ref_embeddings / norms
    sims = anti_norm @ emb_norm
    return float((1 - sims).min())


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


def _filter_videos(videos: dict, features_root, exclude_folders, folder_filter, images_root):
    """Apply exclusion and folder_filter to a videos dict. Returns filtered dict or None on error."""
    if images_root is not None:
        allowed = {d.name for d in images_root.rglob("*") if d.is_dir()}
        videos = {name: paths for name, paths in videos.items() if name in allowed}
        print(f"Restricting to {len(videos)} videos found in {images_root}")

    if exclude_folders:
        for folder in exclude_folders:
            if folder in videos:
                del videos[folder]
                print(f"Excluded: {folder}")

    if folder_filter:
        folder_filter_lower = folder_filter.lower()
        filtered_videos = {
            name: paths for name, paths in videos.items()
            if folder_filter_lower in name.lower()
            or any(folder_filter_lower in p.parent.parent.name.lower() for p in paths[:1])
        }

        if not filtered_videos:
            print(f"\nNo folders found matching '{folder_filter}'")
            print("\nAvailable folders:")
            for name in sorted(videos.keys())[:20]:
                print(f"    - '{name}'")
            if len(videos) > 20:
                print(f"    ... and {len(videos) - 20} more")
            return None

        print(f"\nMatched {len(filtered_videos)} folder(s) for '{folder_filter}':")
        for name in sorted(filtered_videos.keys()):
            print(f"    - '{name}'")
        print()

        videos = filtered_videos

    return videos


def _analyze_video(video_name, feature_paths, window_size, min_gap):
    """Analyze a single video. Returns (changes, scores, features) or None."""
    if len(feature_paths) < 2 * window_size:
        return None

    features = load_video_features(feature_paths)
    if features is None:
        return None

    change_scores = compute_temporal_changes(features, window_size)
    if len(change_scores) == 0:
        return [], change_scores.tolist(), features

    positions = np.argsort(change_scores)[::-1]
    selected = []
    changes = []
    for pos in positions:
        actual_pos = pos + window_size
        too_close = any(abs(actual_pos - s) < min_gap for s in selected)
        if not too_close:
            selected.append(actual_pos)
            changes.append(TemporalChange(
                position=actual_pos,
                change_score=change_scores[pos],
                video_name=video_name,
                frame_paths=feature_paths,
                window_size=window_size,
            ))

    return changes, change_scores.tolist(), features


def find_all_temporal_changes(
        features_root: Path,
        window_size: int = 10,
        min_gap: int = 20,
        exclude_folders: list[str] | None = None,
        folder_filter: str | None = None,
        images_root: Path | None = None,
        _cache: dict | None = None,
        dirty_videos: set[str] | None = None,
) -> tuple[list[TemporalChange], np.ndarray, np.ndarray, np.ndarray]:
    """
    Find ALL temporal discontinuities across all videos.

    Incremental mode: if _cache and dirty_videos are provided, only reprocess
    the dirty videos and reuse cached results for the rest.

    Args:
        features_root: Root directory containing feature files
        window_size: Number of frames before/after to compare
        min_gap: Minimum gap between detected changes
        exclude_folders: List of folder names to exclude
        folder_filter: If provided, only process folders containing this string (case-insensitive)
        images_root: If provided, only process videos with matching image dirs
        _cache: Dict of per-video cached results {video_name: (changes, scores, embeddings)}
                Pass the same dict across calls to enable incremental mode.
        dirty_videos: Set of video names that need reprocessing. If None, process all.

    Returns:
        - List of TemporalChange objects (local maxima with min_gap)
        - Array of ALL change scores for distribution plotting
        - Global mean embedding (across all frames)
        - Global std embedding (across all frames)
    """
    incremental = _cache is not None and dirty_videos is not None

    videos = get_feature_paths_by_video(features_root)
    videos = _filter_videos(videos, features_root, exclude_folders, folder_filter, images_root)
    if videos is None:
        return [], np.array([]), np.array([]), np.array([])

    if incremental:
        to_process = {n: p for n, p in videos.items() if n in dirty_videos}
        cached_count = len(videos) - len(to_process)
        # Remove stale cache entries for videos no longer in the set
        for k in list(_cache.keys()):
            if k not in videos:
                del _cache[k]
        print(f"\n[incremental] Reprocessing {len(to_process)} dirty videos, reusing {cached_count} cached")
    else:
        to_process = videos
        if _cache is not None:
            _cache.clear()
        print(f"\nProcessing {len(videos)} videos with window_size={window_size}")

    for video_name, feature_paths in tqdm(to_process.items(), desc="Analyzing videos"):
        result = _analyze_video(video_name, feature_paths, window_size, min_gap)
        if result is None:
            if _cache is not None and video_name in _cache:
                del _cache[video_name]
            continue
        if _cache is not None:
            _cache[video_name] = result

    # Assemble results from cache (or from freshly computed data)
    all_changes = []
    all_scores = []
    all_embeddings = []

    source = _cache if _cache is not None else {}
    # If no cache, we need to process everything inline (shouldn't happen in normal flow)
    if not _cache:
        for video_name, feature_paths in videos.items():
            result = _analyze_video(video_name, feature_paths, window_size, min_gap)
            if result is None:
                continue
            changes, scores, features = result
            all_changes.extend(changes)
            all_scores.extend(scores)
            all_embeddings.append(features)
    else:
        for video_name in videos:
            if video_name not in _cache:
                continue
            changes, scores, features = _cache[video_name]
            all_changes.extend(changes)
            all_scores.extend(scores)
            all_embeddings.append(features)

    all_changes.sort(key=lambda x: x.change_score, reverse=True)

    if all_embeddings:
        all_embeddings_stacked = np.vstack(all_embeddings)
        global_mean = np.mean(all_embeddings_stacked, axis=0)
        global_std = np.std(all_embeddings_stacked, axis=0)
        print(f"Computed global stats from {len(all_embeddings_stacked):,} embeddings")
    else:
        global_mean = np.array([])
        global_std = np.array([])

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
        anti_ref_embeddings: np.ndarray | None = None,
        anti_ref_threshold: float = 0.3,
):
    """Display the detected temporal changes - shows current and next frame, picks higher deviation from global mean as reference."""
    if not changes:
        print("No changes to display!")
        return [], []

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
    auto_skipped = 0
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
            chosen_emb = emb_before
        else:
            chosen_idx, chosen_label, chosen_path = frame_after_idx, "AFTER", paths[frame_after_idx]
            chosen_emb = emb_after

        # Auto-skip if too similar to an anti-reference
        if anti_ref_embeddings is not None and len(anti_ref_embeddings) > 0:
            dist = _cosine_dist_to_anti_refs(chosen_emb, anti_ref_embeddings)
            if dist < anti_ref_threshold:
                auto_skipped += 1
                continue

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

    if auto_skipped:
        print(f"[anti-ref] Auto-skipped {auto_skipped} changes (similar to anti-references, threshold={anti_ref_threshold})")
    print(f"{len(pairs)} reference candidates remaining")

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
  #exinfo{{color:#f44;font-size:.85em;font-weight:bold}}
  #hint{{font-size:.8em;color:#555;margin-left:auto}}
  #imgs{{flex:1;display:flex;gap:8px;padding:8px;min-height:0}}
  .slot{{flex:1;display:flex;flex-direction:column;min-height:0}}
  .lbl{{font-size:.75em;color:#888;padding:2px 0 4px}}
  .lbl span{{color:#4f4;font-weight:bold}}
  .slot img{{flex:1;object-fit:contain;width:100%;min-height:0;border-radius:4px;background:#222}}
  .chosen img{{outline:3px solid #4f4;border-radius:4px}}
  .excluded img{{outline:3px solid #f44!important;opacity:.4}}
  .excluded .lbl{{color:#f44}}
  .missing{{flex:1;display:flex;align-items:center;justify-content:center;color:#555;font-size:.9em;background:#1a1a1a;border-radius:4px}}
</style></head><body>
<div id="bar">
  <span id="counter"></span>
  <span id="score"></span>
  <span id="exinfo"></span>
  <span id="video"></span>
  <span id="hint">a/d navigate &nbsp; x = exclude &nbsp; Ctrl+C done</span>
</div>
<div id="imgs">
  <div class="slot" id="slotB"><div class="lbl" id="lblB">BEFORE</div><img id="imgB"></div>
  <div class="slot" id="slotA"><div class="lbl" id="lblA">AFTER</div><img id="imgA"></div>
</div>
<script>
var pairs={pairs_json};
var i=0;
var excluded=new Set();
function show(){{
  var p=pairs[i];
  var isEx=excluded.has(i);
  var exCount=excluded.size;
  document.getElementById('counter').textContent=(i+1)+'/'+pairs.length+(isEx?' [EXCLUDED]':'');
  document.getElementById('score').textContent='score='+p.score;
  document.getElementById('exinfo').textContent=exCount?exCount+' excluded':'';
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
  document.getElementById('slotB').className='slot'+(chB?' chosen':'')+(isEx?' excluded':'');
  document.getElementById('slotA').className='slot'+(!chB?' chosen':'')+(isEx?' excluded':'');
  document.getElementById('lblB').innerHTML='BEFORE &mdash; dev='+p.dev_before+(chB?' <span>\\u2713 reference</span>':'');
  document.getElementById('lblA').innerHTML='AFTER &mdash; dev='+p.dev_after+(!chB?' <span>\\u2713 reference</span>':'');
}}
document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='ArrowRight'){{if(i<pairs.length-1){{i++;show();}}}}
  else if(e.key==='a'||e.key==='ArrowLeft'){{if(i>0){{i--;show();}}}}
  else if(e.key==='x'){{
    if(excluded.has(i))excluded.delete(i);else excluded.add(i);
    fetch('/exclude',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{index:i,exclude:excluded.has(i)}})}});
    show();
  }}
}});
show();
</script>
</body></html>"""

    excluded_set = set()

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
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/exclude":
                    data = _json.loads(body)
                    idx = int(data["index"])
                    if data["exclude"]:
                        excluded_set.add(idx)
                    else:
                        excluded_set.discard(idx)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')
                else:
                    self.send_error(404)

        server = HTTPServer(("0.0.0.0", 8767), Handler)
        print(f"\nViewer running at  http://localhost:8767")
        print(f"SSH tunnel:        ssh -L 8767:localhost:8767 user@host")
        print("Press x to mark/unmark as anti-reference, Ctrl+C when done.\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

    # Split into kept references and newly excluded anti-references
    excluded_paths = [ref_feature_paths[i] for i in sorted(excluded_set) if i < len(ref_feature_paths)]
    kept_paths = [p for i, p in enumerate(ref_feature_paths) if i not in excluded_set]

    if excluded_paths:
        print(f"[anti-ref] User excluded {len(excluded_paths)} references")

    return kept_paths, excluded_paths


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
                _kept, _excl = display_temporal_changes(
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