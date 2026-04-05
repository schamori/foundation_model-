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

    results.sort(key=lambda x: x["min_distance"], reverse=True)

    return results, min_distances


def display_similar_images(
        similar_results: list,
        reference_feature_paths: list[Path],
        features_root: Path,
        images_root: Path,
        ref_features_root: Path,
        ref_images_root: Path,
        max_display: int = 10000,
        port: int = 8766,
) -> set[int]:
    """Display images to be filtered via a local web server with a/d keyboard navigation.

    Press x to exclude the reference shown for the current pair.
    Returns set of excluded reference indices (closest_ref_idx values).
    """
    import json as _json
    import urllib.parse
    from http.server import HTTPServer, BaseHTTPRequestHandler

    if not similar_results:
        print("No similar images found below threshold!")
        return set()

    n_display = min(len(similar_results), max_display)

    pairs = []
    for result in similar_results[:n_display]:
        t = feature_path_to_image_path(result["feature_path"], features_root, images_root)
        r = feature_path_to_image_path(
            reference_feature_paths[result["closest_ref_idx"]], ref_features_root, ref_images_root
        )
        pairs.append({
            "t": str(t), "r": str(r), "d": round(float(result["min_distance"]), 4),
            "ri": int(result["closest_ref_idx"]),
        })

    pairs_json = _json.dumps(pairs)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Filtered images</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column}}
  #bar{{padding:10px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}}
  #counter{{font-weight:bold;font-size:1.1em}}
  #dist{{color:#aaa;font-size:.95em}}
  #exinfo{{color:#f44;font-size:.85em;font-weight:bold}}
  #name{{color:#666;font-size:.8em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:40%}}
  #hint{{font-size:.8em;color:#555;margin-left:auto}}
  #imgs{{flex:1;display:flex;gap:8px;padding:8px;min-height:0}}
  .slot{{flex:1;display:flex;flex-direction:column;min-height:0}}
  .lbl{{font-size:.75em;color:#888;padding:2px 0 4px}}
  .slot img{{flex:1;object-fit:contain;width:100%;min-height:0;border-radius:4px;background:#222}}
  .ref-excluded img{{outline:3px solid #f44;opacity:.4}}
  .ref-excluded .lbl{{color:#f44}}
</style></head><body>
<div id="bar">
  <span id="counter"></span>
  <span id="dist"></span>
  <span id="exinfo"></span>
  <span id="name"></span>
  <span id="hint">a/d navigate &nbsp; x = exclude reference &nbsp; Ctrl+C done</span>
</div>
<div id="imgs">
  <div class="slot"><div class="lbl">To be filtered</div><img id="imgT"></div>
  <div class="slot" id="slotR"><div class="lbl" id="lblR">Closest reference</div><img id="imgR"></div>
</div>
<script>
var pairs={pairs_json};
var i=0;
var excludedRefs=new Set();
function show(){{
  var p=pairs[i];
  var isEx=excludedRefs.has(p.ri);
  document.getElementById('counter').textContent=(i+1)+'/'+pairs.length;
  document.getElementById('dist').textContent='dist='+p.d;
  document.getElementById('exinfo').textContent=excludedRefs.size?excludedRefs.size+' refs excluded':'';
  document.getElementById('name').textContent=p.t.split('/').slice(-2).join('/');
  document.getElementById('imgT').src='/img?p='+encodeURIComponent(p.t)+'&i='+i;
  document.getElementById('imgR').src='/img?p='+encodeURIComponent(p.r)+'&i='+i;
  document.getElementById('slotR').className='slot'+(isEx?' ref-excluded':'');
  document.getElementById('lblR').textContent='Closest reference'+(isEx?' [EXCLUDED]':'');
}}
document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='ArrowRight'){{if(i<pairs.length-1){{i++;show();}}}}
  else if(e.key==='a'||e.key==='ArrowLeft'){{if(i>0){{i--;show();}}}}
  else if(e.key==='x'){{
    var ri=pairs[i].ri;
    if(excludedRefs.has(ri))excludedRefs.delete(ri);else excludedRefs.add(ri);
    fetch('/exclude_ref',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{ref_idx:ri,exclude:excludedRefs.has(ri)}})}});
    show();
  }}
}});
show();
</script>
</body></html>"""

    excluded_refs = set()

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
            if self.path == "/exclude_ref":
                data = _json.loads(body)
                ri = int(data["ref_idx"])
                if data["exclude"]:
                    excluded_refs.add(ri)
                else:
                    excluded_refs.discard(ri)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            else:
                self.send_error(404)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nViewer running at  http://localhost:{port}")
    print(f"SSH tunnel:        ssh -L {port}:localhost:{port} user@host")
    print("Press x to exclude a reference, Ctrl+C when done.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    if excluded_refs:
        print(f"[viewer] Excluded {len(excluded_refs)} references")

    return excluded_refs


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
) -> set[str]:
    """Delete filtered images and their corresponding feature files.

    Returns set of affected video folder names (for incremental reprocessing).
    """
    print(f"\nDeleting {len(similar_results)} images...")

    deleted_images = 0
    deleted_features = 0
    affected_videos = set()

    for result in tqdm(similar_results, desc="Deleting"):
        feature_path = result["feature_path"]
        affected_videos.add(feature_path.parent.name)

        # Delete image
        img_path = feature_path_to_image_path(feature_path, features_root, images_root)
        if img_path and img_path.exists():
            os.remove(img_path)
            deleted_images += 1

        # Delete feature file
        if feature_path.exists():
            os.remove(feature_path)
            deleted_features += 1

    print(f"Deleted {deleted_images} images and {deleted_features} feature files ({len(affected_videos)} videos affected)")
    return affected_videos


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