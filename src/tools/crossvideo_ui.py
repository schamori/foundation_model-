"""
Web UI for cross-video pair mining threshold selection.

Categories (subfolders of embeddings_root like MVD, VS_Retrosigmoid, etc.)
are selected via CLI --categories arg. Factor (beta) is set in the frontend.

Usage:
    python src/tools/crossvideo_ui.py
    python src/tools/crossvideo_ui.py --categories MVD VS_Retrosigmoid
"""

from __future__ import annotations

import json
import random
import sys
import threading
import urllib.parse
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Support running both as module and as script
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT, _find_existing, _EMBEDDINGS_CANDIDATES, _FRAMES_CANDIDATES
from src.data.preprocessing import feature_path_to_image_path

PORT = 8771
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "crossvideo_config.json"


def _discover_categories(embeddings_root: Path) -> list[str]:
    """Find category subfolders (dirs that contain procedure subdirs with .npy files)."""
    cats = []
    if not embeddings_root.exists():
        return cats
    for d in sorted(embeddings_root.iterdir()):
        if not d.is_dir():
            continue
        # Category if it has subdirs containing .npy files
        if any(sd.is_dir() and any(sd.glob("*.npy")) for sd in d.iterdir()):
            cats.append(d.name)
        # Or flat layout: dir itself has .npy files
        elif any(d.glob("*.npy")):
            cats.append(d.name)
    return cats


def _collect_videos(
    embeddings_root: Path,
    categories: list[str],
    exclude_folders: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Collect all video folders (with .npy files) under selected categories."""
    exclude = set(exclude_folders or [])
    videos: dict[str, list[Path]] = {}
    for cat in categories:
        cat_dir = embeddings_root / cat
        if not cat_dir.is_dir():
            print(f"[crossvideo_ui] Warning: category '{cat}' not found")
            continue
        # Check if category dir itself has .npy files (flat layout)
        npys = sorted(cat_dir.glob("*.npy"))
        if npys:
            if cat not in exclude:
                videos[cat] = npys
        else:
            # Nested: category/procedure/*.npy
            for proc_dir in sorted(cat_dir.iterdir()):
                if proc_dir.is_dir() and proc_dir.name not in exclude:
                    npys = sorted(proc_dir.glob("*.npy"))
                    if npys:
                        videos[f"{cat}/{proc_dir.name}"] = npys
    return videos


def _build_pairs(
    videos: dict[str, list[Path]],
    embeddings_root: Path,
    images_root: Path,
    n_samples: int = 200,
) -> list[dict]:
    """Run cross-video pair mining on selected videos."""
    random.seed(42)
    np.random.seed(42)

    feats, paths, labels = [], [], []
    for video_name, vpaths in tqdm(videos.items(), desc="Loading features"):
        for p in vpaths:
            try:
                feats.append(np.load(p))
                paths.append(p)
                labels.append(video_name)
            except Exception:
                pass
    if len(feats) < 2:
        return []

    all_features = np.vstack(feats).astype(np.float32)
    n_total = len(all_features)

    video_indices: dict[str, list[int]] = defaultdict(list)
    global_to_local: dict[int, int] = {}  # global idx → position within its video
    for idx, label in enumerate(labels):
        global_to_local[idx] = len(video_indices[label])
        video_indices[label].append(idx)

    # L2-normalize for cosine sim via inner product
    norms = np.linalg.norm(all_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = all_features / norms

    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False

    # Sample query indices
    if n_samples < n_total:
        valid = [v for v in videos if len(video_indices[v]) > 1]
        sample_indices = []
        for _ in range(n_samples):
            v = random.choice(valid)
            vil = video_indices[v]
            sample_indices.append(vil[random.randint(0, len(vil) - 2)])
    else:
        sample_indices = list(range(n_total))

    # Group sampled queries by video so we build each per-video index once
    queries_by_video: dict[str, list[int]] = defaultdict(list)
    for qi in sample_indices:
        queries_by_video[labels[qi]].append(qi)

    # Global index → index within other-video array (built per query-video)
    results_map: dict[int, dict] = {}

    for qvideo, qis in tqdm(queries_by_video.items(), desc="Mining pairs"):
        q_mask = np.array([l == qvideo for l in labels])
        other_mask = ~q_mask
        other_normed = normed[other_mask]
        other_global = np.where(other_mask)[0]  # global indices of other-video frames

        if len(other_normed) == 0:
            for qi in qis:
                results_map[qi] = {"best_sim": None, "best_global": None}
            continue

        if use_faiss:
            import faiss
            index = faiss.IndexFlatIP(other_normed.shape[1])
            index.add(other_normed)
            q_vecs = normed[qis]
            dists, idxs = index.search(q_vecs, 1)
            for qi, sim, idx in zip(qis, dists[:, 0], idxs[:, 0]):
                results_map[qi] = {
                    "best_sim": float(sim) if idx >= 0 else None,
                    "best_global": int(other_global[idx]) if idx >= 0 else None,
                }
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vecs = normed[qis]
            sims = cosine_similarity(q_vecs, other_normed)
            for qi, sim_row in zip(qis, sims):
                best_local = int(np.argmax(sim_row))
                results_map[qi] = {
                    "best_sim": float(sim_row[best_local]),
                    "best_global": int(other_global[best_local]),
                }

    results = []
    for qi in sample_indices:
        qvideo = labels[qi]
        vil = video_indices[qvideo]
        local_idx = global_to_local[qi]

        next_sim = None
        if local_idx + 1 < len(vil):
            next_sim = float(np.dot(normed[qi], normed[vil[local_idx + 1]]))

        info = results_map.get(qi, {})
        best_sim = info.get("best_sim")
        best_global = info.get("best_global")

        qip = feature_path_to_image_path(paths[qi], embeddings_root, images_root)
        cip = (feature_path_to_image_path(paths[best_global], embeddings_root, images_root)
               if best_global is not None else None)

        results.append({
            "query_path": str(qip) if qip else str(paths[qi]),
            "query_video": qvideo,
            "next_sim": next_sim,
            "cross_sim": best_sim,
            "cross_path": str(cip) if cip else None,
            "cross_video": labels[best_global] if best_global is not None else None,
        })

    return results


def _build_html(categories: list[str], n_videos: int) -> str:
    """Build the cross-video threshold picker HTML."""
    cats_str = ", ".join(categories)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Cross-Video Pair Threshold</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#eee;font-family:sans-serif;display:flex;flex-direction:column;height:100vh}}
#top{{padding:12px 16px;background:#1a1a1a;display:flex;align-items:center;gap:12px;flex-shrink:0;flex-wrap:wrap}}
#top label{{font-size:.9em;color:#aaa}}
input[type=number]{{background:#222;color:#eee;border:1px solid #444;padding:4px 8px;font-size:.9em;border-radius:3px}}
.btn{{border:1px solid #555;padding:6px 14px;cursor:pointer;border-radius:4px;font-size:.9em}}
#runBtn{{background:#2a4a6a;color:#fff;border-color:#4a7aaa}}
#runBtn:hover{{background:#3a6a9a}}
#runBtn:disabled{{opacity:.5;cursor:default}}
#saveBtn{{background:#2a5a2a;color:#fff;border-color:#4a4}}
#saveBtn:hover{{background:#3a7a3a}}
#savedMsg{{color:#4a4;font-size:.85em;display:none}}
#status{{color:#f90;font-size:.85em}}
.info{{color:#888;font-size:.85em}}
#controls{{padding:8px 16px;background:#1a1a1a;border-bottom:1px solid #333;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
#factorSlider{{width:250px}}
#factorVal{{color:#f90;font-weight:bold;min-width:40px}}
#stats{{color:#aaa;font-size:.85em}}
#main{{flex:1;display:flex;min-height:0;overflow:hidden}}
#chartPanel{{width:380px;padding:12px;display:flex;flex-direction:column;border-right:1px solid #333}}
canvas{{background:#1a1a1a;border-radius:4px;flex:1}}
#pairPanel{{flex:1;display:flex;flex-direction:column;padding:8px;min-height:0}}
#pairNav{{display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-shrink:0}}
#pairNav .idx{{color:#f90;font-weight:bold}}
#pairNav .hint{{color:#555;font-size:.75em}}
.pair{{flex:1;background:#1a1a1a;border-radius:6px;padding:12px;display:flex;flex-direction:column;min-height:0}}
.pair-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.pair-match{{color:#4a4;font-weight:bold}}
.pair-nomatch{{color:#a44;font-weight:bold}}
.pair-video{{color:#666;font-size:.75em}}
.pair-sims{{font-size:.85em;color:#888;margin-bottom:8px}}
.pair-imgs{{display:flex;gap:8px;flex:1;min-height:0}}
.pair-imgs div{{flex:1;text-align:center;display:flex;flex-direction:column;min-height:0}}
.pair-imgs img{{width:100%;max-height:100%;object-fit:contain;border-radius:4px;background:#222;flex:1;min-height:0}}
.pair-imgs .lbl{{font-size:.75em;color:#666;margin-top:4px;flex-shrink:0}}
.hidden{{display:none}}
</style></head><body>
<div id="top">
  <span class="info">Categories: <b>{cats_str}</b> ({n_videos} videos)</span>
  <span id="status" style="color:#f90;font-size:.85em">Loading...</span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
    <button class="btn" id="saveBtn" onclick="saveThreshold()">Save threshold</button>
    <span id="savedMsg">Saved!</span>
  </div>
</div>
<div id="controls" class="hidden">
  <label>Factor (beta):</label>
  <input type="range" id="factorSlider" min="1.0" max="20.0" step="0.1" value="3.0">
  <span id="factorVal">3.0</span>
  <span id="stats"></span>
</div>
<div id="main">
  <div id="chartPanel"><canvas id="chart"></canvas></div>
  <div id="pairPanel"></div>
</div>
<script>
var rawData=null, sortedPairs=[], pairIdx=0;
var canvas=document.getElementById('chart');
var ctx=canvas.getContext('2d');

fetch('/api/results')
  .then(r=>r.json()).then(rd=>{{
    rawData=rd.results;
    document.getElementById('status').textContent=rawData.length+' pairs loaded';
    document.getElementById('controls').classList.remove('hidden');
    refilter();
  }});

function refilter(){{
  if(!rawData)return;
  var factor=parseFloat(document.getElementById('factorSlider').value);
  document.getElementById('factorVal').textContent=factor.toFixed(1);

  var data=rawData.map(r=>{{
    var isMatch=false;
    if(r.next_sim!=null&&r.cross_sim!=null) isMatch=r.cross_sim>(r.next_sim/factor);
    var margin=isMatch?(r.cross_sim-(r.next_sim/factor)):null;
    return Object.assign({{}},r,{{is_match:isMatch,margin:margin}});
  }});

  var matches=data.filter(r=>r.is_match);
  var pct=data.length?((matches.length/data.length)*100).toFixed(1):'0';
  document.getElementById('stats').textContent=
    matches.length+'/'+data.length+' matches ('+pct+'%)';

  // Sort by worst margin first (smallest positive margin = closest to threshold)
  sortedPairs=[...matches].sort((a,b)=>a.margin-b.margin);
  pairIdx=0;

  drawChart(data,factor);
  showPair();
}}

function drawChart(data,factor){{
  var dpr=window.devicePixelRatio||1;
  var w=canvas.clientWidth,h=canvas.clientHeight;
  canvas.width=w*dpr;canvas.height=h*dpr;
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  var pad=40,pw=w-pad*2,ph=h-pad*2;

  var allNext=data.filter(r=>r.next_sim!=null).map(r=>r.next_sim);
  var allCross=data.filter(r=>r.cross_sim!=null).map(r=>r.cross_sim);
  if(!allNext.length)return;
  var mnN=Math.min(...allNext),mxN=Math.max(...allNext);
  var mnC=Math.min(...allCross),mxC=Math.max(...allCross);
  if(mxN==mnN)mxN=mnN+0.01;
  if(mxC==mnC)mxC=mnC+0.01;

  ctx.strokeStyle='#888';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  var y0=pad+ph-((mnN/factor-mnC)/(mxC-mnC))*ph;
  var y1=pad+ph-((mxN/factor-mnC)/(mxC-mnC))*ph;
  ctx.beginPath();ctx.moveTo(pad,y0);ctx.lineTo(pad+pw,y1);ctx.stroke();
  ctx.setLineDash([]);

  data.forEach(r=>{{
    if(r.next_sim==null||r.cross_sim==null)return;
    var x=pad+((r.next_sim-mnN)/(mxN-mnN))*pw;
    var y=pad+ph-((r.cross_sim-mnC)/(mxC-mnC))*ph;
    ctx.fillStyle=r.is_match?'rgba(100,200,100,0.7)':'rgba(200,100,100,0.5)';
    ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
  }});

  ctx.fillStyle='#888';ctx.font='11px sans-serif';
  ctx.fillText('Next frame sim',w/2-35,h-5);
  ctx.save();ctx.translate(12,h/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('Cross-video sim',0,0);ctx.restore();
}}

function showPair(){{
  var panel=document.getElementById('pairPanel');
  if(!sortedPairs.length){{panel.innerHTML='<div style="color:#666;padding:20px">No matches</div>';return;}}
  var r=sortedPairs[pairIdx];
  var factor=parseFloat(document.getElementById('factorSlider').value);
  var html='<div id="pairNav"><span class="idx">'+(pairIdx+1)+' / '+sortedPairs.length+'</span>';
  html+='<span class="hint">A/D or Left/Right to navigate</span>';
  html+='<span style="color:#888;font-size:.8em">margin: '+r.margin.toFixed(4)+'</span></div>';
  html+='<div class="pair">';
  html+='<div class="pair-header"><span class="pair-match">MATCH</span>';
  html+='<span class="pair-video">'+r.query_video+'</span></div>';
  html+='<div class="pair-sims">next='+((r.next_sim||0).toFixed(4))+' cross='+((r.cross_sim||0).toFixed(4));
  html+=' thresh='+((r.next_sim||0)/factor).toFixed(4)+'</div>';
  html+='<div class="pair-imgs">';
  var qimg='/img?p='+encodeURIComponent(r.query_path);
  html+='<div><img src="'+qimg+'"><div class="lbl">Query</div></div>';
  if(r.cross_path){{var cimg='/img?p='+encodeURIComponent(r.cross_path);
  html+='<div><img src="'+cimg+'"><div class="lbl">Cross ('+((r.cross_video||'').substring(0,25))+')</div></div>';}}
  html+='</div></div>';
  panel.innerHTML=html;
}}

document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='D'||e.key==='ArrowRight'){{
    if(pairIdx<sortedPairs.length-1){{pairIdx++;showPair();}}
  }}else if(e.key==='a'||e.key==='A'||e.key==='ArrowLeft'){{
    if(pairIdx>0){{pairIdx--;showPair();}}
  }}
}});

document.getElementById('factorSlider').addEventListener('input',refilter);

function saveThreshold(){{
  if(!rawData){{alert('Compute pairs first');return;}}
  var factor=parseFloat(document.getElementById('factorSlider').value);
  fetch('/api/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{factor:factor}})}}).then(r=>r.json()).then(d=>{{
    if(d.ok){{document.getElementById('savedMsg').style.display='inline';
      setTimeout(()=>document.getElementById('savedMsg').style.display='none',2000);}}
  }});
}}

window.addEventListener('resize',()=>{{if(rawData)refilter();}});
</script></body></html>"""


def run_crossvideo_ui(
    embeddings_root: Path,
    images_root: Path | None = None,
    categories: list[str] | None = None,
    n_samples: int = 200,
    exclude_folders: list[str] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    port: int = PORT,
):
    """Launch the cross-video threshold picker web UI."""
    if images_root is None:
        images_root = _find_existing(_FRAMES_CANDIDATES, embeddings_root.parent / "Extracted Frames")

    # Auto-discover categories if not specified
    if categories is None:
        categories = _discover_categories(embeddings_root)

    videos = _collect_videos(embeddings_root, categories, exclude_folders)
    print(f"[crossvideo_ui] Categories: {categories}")
    print(f"[crossvideo_ui] {len(videos)} videos total")

    # Compute all pairs BEFORE starting server
    print(f"[crossvideo_ui] Computing ALL pairs for {len(videos)} videos...")
    all_results = _build_pairs(videos, embeddings_root, images_root, n_samples=999_999_999)

    valid_results = [r for r in all_results if r.get("cross_sim") is not None]

    print(f"[crossvideo_ui] Done: {len(all_results)} pairs. Sending {len(valid_results)} valid pairs to UI.")

    html = _build_html(categories, len(videos))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json_response(self, data, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path == "/api/results":
                self._json_response({"results": valid_results})
            elif self.path.startswith("/img?"):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                p = Path(qs.get("p", [""])[0])
                if p.exists():
                    self.send_response(200)
                    ct = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    self.send_header("Content-Type", ct)
                    self.end_headers()
                    self.wfile.write(p.read_bytes())
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            d = json.loads(body) if body else {}

            if self.path == "/api/save":
                factor = d["factor"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump({
                        "cross_video_factor": factor,
                        "categories": categories,
                    }, f, indent=2)
                print(f"[crossvideo_ui] Saved factor={factor} -> {output_path}")
                self._json_response({"ok": True})
            else:
                self.send_error(404)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n[crossvideo_ui] http://localhost:{port}")
    print("[crossvideo_ui] Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


def load_crossvideo_threshold(path: Path = DEFAULT_OUTPUT) -> float | None:
    """Load saved factor (for use in training without launching UI)."""
    if path.exists():
        with open(path) as f:
            return json.load(f).get("cross_video_factor")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    default_emb = _find_existing(_EMBEDDINGS_CANDIDATES, _EMBEDDINGS_CANDIDATES[0])
    parser = argparse.ArgumentParser(description="Cross-video pair threshold picker")
    parser.add_argument("--embeddings-root", type=Path, default=default_emb)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--categories", nargs="*", default=["MVD"],
                        help="Category folders to include (e.g. MVD VS_Retrosigmoid). "
                             "Default: all categories found.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--exclude", nargs="*", default=["reference for filtering", "reference images"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    run_crossvideo_ui(
        embeddings_root=args.embeddings_root,        images_root=args.images_root,
        categories=args.categories,
        n_samples=args.n_samples,
        exclude_folders=args.exclude,
        output_path=args.output,
        port=args.port,
    )


if __name__ == "__main__":
    main()

