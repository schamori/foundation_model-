"""
Complete pipeline: Find reference images → Extract features → Filter dataset
"""

from pathlib import Path
import shutil
import json
import numpy as np
from datetime import datetime

# Import from existing modules
from find_reference_for_cleanup import (
    find_all_temporal_changes,
    display_temporal_changes,
    plot_full_distribution,
    plot_filtered_distribution,
    load_anti_references,
    save_anti_references,
)

from filter_images import (
    get_feature_paths,
    load_features,
    find_similar_images,
    plot_distance_histogram,
    delete_filtered_images,
    display_similar_images,
    feature_path_to_image_path,
)


def save_run_config(config_dir: Path, temporal_threshold: float, filter_threshold: float,
                    folder_filter: str | None, n_references: int, n_deleted: int):
    """Save run config to saved_configs/ with timestamp."""
    config_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "timestamp": ts,
        "temporal_threshold": temporal_threshold,
        "filter_threshold": filter_threshold,
        "folder_filter": folder_filter,
        "n_references": n_references,
        "n_deleted": n_deleted,
    }
    p = config_dir / f"run_{ts}.json"
    with open(p, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved run config to {p}")


def check_frame_distances(features_root: Path, images_root: Path, checks: dict[str, list[int]]):
    """Show a web viewer for frame pairs with their cosine distance.

    checks: {video_name: [frame_a, frame_b, ...]} — shows each consecutive pair.
    """
    import json as _json
    import urllib.parse
    from http.server import HTTPServer, BaseHTTPRequestHandler

    pairs = []
    for video_name, frames in checks.items():
        candidates = []
        for d in features_root.rglob("*"):
            if d.is_dir() and (d.name == video_name or d.name.startswith(video_name)):
                candidates.append(d)
        if not candidates:
            print(f"[check] {video_name}: no embedding directory found")
            continue
        vdir = sorted(candidates)[0]
        print(f"[check] {video_name} → {vdir.name}")

        for i in range(len(frames) - 1):
            fa, fb = frames[i], frames[i + 1]
            pa = list(vdir.glob(f"*{fa:06d}.npy")) or list(vdir.glob(f"*{fa}.npy"))
            pb = list(vdir.glob(f"*{fb:06d}.npy")) or list(vdir.glob(f"*{fb}.npy"))
            if not pa or not pb:
                print(f"  frames {fa}-{fb}: embedding not found (a={len(pa)}, b={len(pb)})")
                continue
            ea = np.load(pa[0]).flatten()
            eb = np.load(pb[0]).flatten()
            ea_n = ea / (np.linalg.norm(ea) + 1e-8)
            eb_n = eb / (np.linalg.norm(eb) + 1e-8)
            cos_dist = float(1 - np.dot(ea_n, eb_n))
            print(f"  frames {fa}-{fb}: cosine_dist = {cos_dist:.6f}")

            img_a = feature_path_to_image_path(pa[0], features_root, images_root)
            img_b = feature_path_to_image_path(pb[0], features_root, images_root)
            pairs.append({
                "before": str(img_a) if img_a and img_a.exists() else "",
                "after": str(img_b) if img_b and img_b.exists() else "",
                "score": round(cos_dist, 6),
                "video": video_name,
                "frame_a": fa,
                "frame_b": fb,
            })

    if not pairs:
        print("[check] No valid pairs to show")
        return

    pairs_json = _json.dumps(pairs)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Frame distance check</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column}}
  #bar{{padding:10px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}}
  #counter{{font-weight:bold;font-size:1.1em}}
  #score{{color:#f90;font-size:1.1em;font-weight:bold}}
  #video{{color:#666;font-size:.8em}}
  #hint{{font-size:.8em;color:#555;margin-left:auto}}
  #imgs{{flex:1;display:flex;gap:8px;padding:8px;min-height:0}}
  .slot{{flex:1;display:flex;flex-direction:column;min-height:0}}
  .lbl{{font-size:.75em;color:#888;padding:2px 0 4px}}
  .slot img{{flex:1;object-fit:contain;width:100%;min-height:0;border-radius:4px;background:#222}}
</style></head><body>
<div id="bar">
  <span id="counter"></span>
  <span id="score"></span>
  <span id="video"></span>
  <span id="hint">a/d or arrows to navigate &nbsp; Ctrl+C when done</span>
</div>
<div id="imgs">
  <div class="slot"><div class="lbl" id="lblA"></div><img id="imgA"></div>
  <div class="slot"><div class="lbl" id="lblB"></div><img id="imgB"></div>
</div>
<script>
var pairs={pairs_json};
var i=0;
function show(){{
  var p=pairs[i];
  document.getElementById('counter').textContent=(i+1)+'/'+pairs.length;
  document.getElementById('score').textContent='cosine_dist = '+p.score.toFixed(6);
  document.getElementById('video').textContent=p.video;
  document.getElementById('lblA').textContent='Frame '+p.frame_a;
  document.getElementById('lblB').textContent='Frame '+p.frame_b;
  document.getElementById('imgA').src=p.before?'/img?p='+encodeURIComponent(p.before):'';
  document.getElementById('imgB').src=p.after?'/img?p='+encodeURIComponent(p.after):'';
}}
document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='ArrowRight'){{if(i<pairs.length-1){{i++;show();}}}}
  else if(e.key==='a'||e.key==='ArrowLeft'){{if(i>0){{i--;show();}}}}
}});
show();
</script>
</body></html>"""

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

    server = HTTPServer(("0.0.0.0", 8768), Handler)
    print(f"\n[check] Viewer at http://localhost:8768  ({len(pairs)} pairs)")
    print("[check] Ctrl+C to continue to main pipeline\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def load_latest_config(saved_configs_dir: Path) -> dict | None:
    """Load the most recent saved run config."""
    if not saved_configs_dir.exists():
        return None
    configs = sorted(saved_configs_dir.glob("run_*.json"))
    if not configs:
        return None
    with open(configs[-1]) as f:
        cfg = json.load(f)
    print(f"[continue] Loaded config from {configs[-1].name}: "
          f"temporal={cfg['temporal_threshold']}, filter={cfg['filter_threshold']}, "
          f"folder_filter={cfg.get('folder_filter')}")
    return cfg


def main():
    # ===== CONFIGURATION =====
    CONTINUE = False  # Set True to load last config and run without prompts

    images_root = Path(r"/media/HDD1/moritz/Extracted Frames")
    features_root = Path(r"/media/HDD1/moritz/Extracted Frames embeddings")
    reference_images_dir = images_root / "reference images"
    reference_features_dir = features_root / "reference images"

    exclude_folders = ["reference for filtering", "reference images"]
    folder_filter =  None

    # Temporal detection params
    window_size = 1
    min_gap = 1
    max_display = 99999

    # Filter threshold
    filter_threshold = 0.1

    # Anti-reference config (per folder_filter)
    anti_ref_dir = images_root / "anti_references"
    anti_ref_file = anti_ref_dir / f"{folder_filter}.json" if folder_filter else anti_ref_dir / "default.json"
    anti_ref_threshold = 0.2
    saved_configs_dir = Path(__file__).resolve().parent / "saved_configs"
    # =========================

    # Continue mode: load last config and skip all prompts
    if CONTINUE:
        cfg = load_latest_config(saved_configs_dir)
        if cfg is None:
            print("[continue] No saved config found — falling back to interactive mode")
            CONTINUE = False
        else:
            filter_threshold = cfg["filter_threshold"]
            folder_filter = cfg.get("folder_filter")
            anti_ref_file = anti_ref_dir / f"{folder_filter}.json" if folder_filter else anti_ref_dir / "default.json"
            saved_threshold = cfg["temporal_threshold"]

    # Debug: check specific frame pair distances
    debug_checks = {
        # "5ALA_006_08102025_21636994_Op02_R-T": [3640, 3815],
        # "ATLR_31": [6372 , 6374  ]

    }
    if debug_checks and not CONTINUE:
        check_frame_distances(features_root, images_root, debug_checks)

    if CONTINUE:
        loop_through = True
        first_run = False
    else:
        loop_through = input("Loop through automatically after first run? (y/n): ").strip().lower() == 'y'
        first_run = True
        saved_threshold = None

    # Load persistent anti-references
    anti_ref_paths, anti_ref_embeddings = load_anti_references(anti_ref_file)
    if anti_ref_paths:
        print(f"[anti-ref] Loaded {len(anti_ref_paths)} anti-references from {anti_ref_file}")

    # Incremental caches — persist across loop iterations
    temporal_cache = {}   # {video_name: (changes, scores, features)} for find_all_temporal_changes
    dirty_videos = None   # None = full run on first iteration, set(...) = incremental
    cached_target_features = None   # np.ndarray
    cached_target_paths = None      # list[Path]

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
            images_root=images_root,
            folder_filter=folder_filter,
            _cache=temporal_cache,
            dirty_videos=dirty_videos,
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

        if first_run:
            filtered_changes = []
            while True:
                filtered_changes = [c for c in all_changes if c.change_score >= threshold]

                if not filtered_changes:
                    print(f"No changes found >= {threshold}. Dataset is clean!")
                    break

                plot_filtered_distribution(all_changes, threshold)
                print(f"\n{len(filtered_changes)} changes >= {threshold}")

                ref_feature_paths, newly_excluded = display_temporal_changes(
                    filtered_changes[:max_display],
                    features_root,
                    images_root,
                    global_mean,
                    global_std,
                    anti_ref_embeddings=anti_ref_embeddings,
                    anti_ref_threshold=anti_ref_threshold,
                )

                # Save any new exclusions to anti-reference file
                if newly_excluded:
                    anti_ref_paths.extend(newly_excluded)
                    save_anti_references(anti_ref_file, anti_ref_paths)
                    # Reload embeddings with new additions
                    anti_ref_paths, anti_ref_embeddings = load_anti_references(anti_ref_file)

                print(f"Using {len(ref_feature_paths)} reference feature files from existing embeddings")

                t = input(f"\nEnter new temporal threshold to retry (current={threshold}), or press Enter to proceed to step 2: ").strip()
                if t:
                    try:
                        threshold = float(t)
                        saved_threshold = threshold
                    except ValueError:
                        print("Invalid input, keeping current threshold.")
                else:
                    break

            if not filtered_changes:
                break
        else:
            filtered_changes = [c for c in all_changes if c.change_score >= threshold]

            if not filtered_changes:
                print(f"No changes found >= {threshold}. Dataset is clean!")
                break

            print(f"\n{len(filtered_changes)} changes >= {threshold}")

            ref_feature_paths, newly_excluded = display_temporal_changes(
                filtered_changes[:max_display],
                features_root,
                images_root,
                global_mean,
                global_std,
                show_viewer=False,
                anti_ref_embeddings=anti_ref_embeddings,
                anti_ref_threshold=anti_ref_threshold,
            )

            # Save any new exclusions (shouldn't happen in auto mode, but just in case)
            if newly_excluded:
                anti_ref_paths.extend(newly_excluded)
                save_anti_references(anti_ref_file, anti_ref_paths)
                anti_ref_paths, anti_ref_embeddings = load_anti_references(anti_ref_file)

            print(f"Using {len(ref_feature_paths)} reference feature files from existing embeddings")

        if not ref_feature_paths:
            print("\nNo reference features to filter with (all were anti-ref excluded). Lowering threshold or clearing anti_references.json may help.")
            break

        # ==================== STEP 2: Filter dataset ====================
        print("\n" + "=" * 60)
        print("STEP 2: FILTER DATASET")
        print("=" * 60)

        ref_features, ref_feature_paths = load_features(ref_feature_paths)

        # Incremental: if we have cached targets, just remove deleted paths
        if cached_target_features is not None and dirty_videos is not None:
            deleted_paths = {str(p) for p in cached_target_paths
                            if p.parent.name in dirty_videos and not p.exists()}
            if deleted_paths:
                keep = [i for i, p in enumerate(cached_target_paths) if str(p) not in deleted_paths]
                cached_target_features = cached_target_features[keep]
                cached_target_paths = [cached_target_paths[i] for i in keep]
                print(f"[incremental] Removed {len(deleted_paths)} deleted features from cache "
                      f"({len(cached_target_paths)} remaining)")
            target_features = cached_target_features
            target_feature_paths = cached_target_paths
        else:
            # Full load on first iteration
            allowed_videos = {d.name for d in images_root.rglob("*") if d.is_dir()}
            target_feature_paths = [p for p in get_feature_paths(features_root, exclude_folders=exclude_folders)
                                     if p.parent.name in allowed_videos]
            target_features, target_feature_paths = load_features(target_feature_paths)
            cached_target_features = target_features
            cached_target_paths = target_feature_paths

        if first_run:
            similar_results = []
            while True:
                similar_results, min_distances = find_similar_images(
                    ref_features, target_features, target_feature_paths, filter_threshold
                )
                print(f"\nFound {len(similar_results)} images to filter")
                plot_distance_histogram(min_distances, filter_threshold)

                if not similar_results:
                    print("No similar images found. Dataset is clean!")
                    break

                if input(f"\nDisplay {len(similar_results)} images to be deleted? (y/n): ").strip().lower() == 'y':
                    excluded_ref_idxs = display_similar_images(
                        similar_results,
                        ref_feature_paths,
                        features_root=features_root,
                        images_root=images_root,
                        ref_features_root=features_root,
                        ref_images_root=images_root,
                        max_display=len(similar_results),
                    )
                    if excluded_ref_idxs:
                        # Remove excluded references and recompute
                        keep = [i for i in range(len(ref_feature_paths)) if i not in excluded_ref_idxs]
                        ref_feature_paths = [ref_feature_paths[i] for i in keep]
                        ref_features = ref_features[keep]
                        print(f"[viewer] Removed {len(excluded_ref_idxs)} references, {len(ref_feature_paths)} remaining — recomputing...")
                        continue  # recompute similar_results with updated refs

                t = input(f"\nEnter new filter threshold to retry (current={filter_threshold}), or press Enter to proceed: ").strip()
                if t:
                    try:
                        filter_threshold = float(t)
                    except ValueError:
                        print("Invalid input, keeping current threshold.")
                else:
                    break

            if not similar_results:
                break

            # Delete
            if input(f"\nDELETE {len(similar_results)} images? (yes/no): ").strip().lower() == 'yes':
                dirty_videos = delete_filtered_images(similar_results, features_root, images_root)
                save_run_config(saved_configs_dir, threshold, filter_threshold,
                                folder_filter, len(ref_feature_paths), len(similar_results))
                print("\nRestarting pipeline to find more...\n")
                first_run = True if not CONTINUE else not loop_through
            else:
                break
        else:
            similar_results, min_distances = find_similar_images(
                ref_features, target_features, target_feature_paths, filter_threshold
            )
            print(f"\nFound {len(similar_results)} images to filter")

            if not similar_results:
                print("No similar images found. Dataset is clean!")
                break

            # Auto mode - just delete
            dirty_videos = delete_filtered_images(similar_results, features_root, images_root)
            save_run_config(saved_configs_dir, threshold, filter_threshold,
                            folder_filter, len(ref_feature_paths), len(similar_results))
            print("\nAuto-looping to find more...\n")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
