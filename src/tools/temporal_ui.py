"""
Web UI for temporal discontinuity threshold selection.

When run as __main__: opens browser UI to interactively pick threshold.
When imported: use compute_and_get_threshold() to load/return saved threshold.

Usage:
    python -m src.tools.temporal_ui --embeddings-root /path/to/embeddings
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import numpy as np

# Support running both as `python -m src.tools.temporal_ui` and as a script
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT, _find_existing, _EMBEDDINGS_CANDIDATES, _FRAMES_CANDIDATES
from src.data.preprocessing import (
    compute_all_temporal_scores,
    feature_path_to_image_path,
    get_feature_paths_by_video,
    load_video_features,
    save_temporal_scores,
)

PORT = 8770
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "temporal_config.json"


def _build_html(all_scores: list[float], changes: list[dict], images_root: str, features_root: str) -> str:
    """Build the interactive threshold picker HTML."""
    scores_json = json.dumps(all_scores)
    changes_json = json.dumps(changes[:500])  # limit for browser performance
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Temporal Discontinuity Threshold</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#eee;font-family:sans-serif;display:flex;flex-direction:column;height:100vh}}
#top{{padding:12px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}}
#top label{{font-size:.9em;color:#aaa}}
#threshold{{width:200px}}
#threshVal{{color:#f90;font-weight:bold;font-size:1.1em;min-width:80px}}
#stats{{font-size:.85em;color:#888}}
#saveBtn{{background:#2a5a2a;color:#fff;border:1px solid #4a4;padding:6px 16px;cursor:pointer;border-radius:4px}}
#saveBtn:hover{{background:#3a7a3a}}
#savedMsg{{color:#4a4;font-size:.85em;display:none}}
#main{{flex:1;display:flex;min-height:0}}
#histPanel{{flex:1;padding:12px;display:flex;flex-direction:column}}
canvas{{background:#1a1a1a;border-radius:4px;width:100%;flex:1}}
#pairPanel{{width:400px;overflow-y:auto;padding:8px;border-left:1px solid #333}}
.pair{{margin-bottom:8px;background:#1a1a1a;border-radius:4px;padding:6px}}
.pair-score{{color:#f90;font-size:.85em;font-weight:bold}}
.pair-video{{color:#666;font-size:.75em}}
.pair-imgs{{display:flex;gap:4px;margin-top:4px}}
.pair-imgs img{{width:50%;border-radius:3px;cursor:pointer}}
</style></head><body>
<div id="top">
  <label>Threshold:</label>
  <input type="range" id="threshold" min="0" max="1" step="0.001" value="0.1">
  <span id="threshVal">0.100</span>
  <span id="stats"></span>
  <button id="saveBtn" onclick="saveThreshold()">Save threshold</button>
  <span id="savedMsg">Saved!</span>
</div>
<div id="main">
  <div id="histPanel"><canvas id="hist"></canvas></div>
  <div id="pairPanel" id="pairs"></div>
</div>
<script>
var allScores={scores_json};
var changes={changes_json};
var imagesRoot="{images_root}";
var featuresRoot="{features_root}";

var canvas=document.getElementById('hist');
var ctx=canvas.getContext('2d');
var slider=document.getElementById('threshold');
var threshEl=document.getElementById('threshVal');
var statsEl=document.getElementById('stats');
var pairPanel=document.getElementById('pairPanel');

// Compute histogram bins
var nBins=100;
var mn=Math.min(...allScores),mx=Math.max(...allScores);
var binW=(mx-mn)/nBins;
var bins=new Array(nBins).fill(0);
allScores.forEach(s=>{{var b=Math.min(Math.floor((s-mn)/binW),nBins-1);bins[b]++;}});
var maxBin=Math.max(...bins);

slider.min=mn.toFixed(4);
slider.max=mx.toFixed(4);
slider.value=((mn+mx)/2).toFixed(4);

function drawHist(){{
  var dpr=window.devicePixelRatio||1;
  var w=canvas.clientWidth,h=canvas.clientHeight;
  canvas.width=w*dpr;canvas.height=h*dpr;
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  var pad=40,pw=w-pad*2,ph=h-pad*2;
  var thresh=parseFloat(slider.value);

  // Bars
  var bw=pw/nBins;
  for(var i=0;i<nBins;i++){{
    var x=pad+i*bw;
    var bh=(bins[i]/maxBin)*ph;
    var val=mn+i*binW;
    ctx.fillStyle=val>=thresh?'rgba(255,170,51,0.8)':'rgba(100,100,200,0.5)';
    ctx.fillRect(x,pad+ph-bh,bw-1,bh);
  }}

  // Threshold line
  var tx=pad+((thresh-mn)/(mx-mn))*pw;
  ctx.strokeStyle='#f00';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(tx,pad);ctx.lineTo(tx,pad+ph);ctx.stroke();

  // Axes
  ctx.strokeStyle='#555';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad+ph);ctx.lineTo(pad+pw,pad+ph);ctx.stroke();
  ctx.fillStyle='#888';ctx.font='11px sans-serif';
  for(var i=0;i<=5;i++){{
    var v=mn+(mx-mn)*i/5;
    var x=pad+(i/5)*pw;
    ctx.fillText(v.toFixed(3),x-15,pad+ph+15);
  }}
  ctx.fillStyle='#aaa';ctx.font='13px sans-serif';
  ctx.fillText('Temporal Change Score',w/2-60,h-5);

  // Stats
  var above=allScores.filter(s=>s>=thresh).length;
  var pct=(above/allScores.length*100).toFixed(1);
  statsEl.textContent=above+' / '+allScores.length+' above ('+pct+'%)';
}}

function updatePairs(){{
  var thresh=parseFloat(slider.value);
  var above=changes.filter(c=>c.score>=thresh).sort((a,b)=>a.score-b.score);
  var html='<div style="color:#888;font-size:.8em;padding:4px">'+above.length+' changes above threshold</div>';
  above.slice(0,30).forEach(c=>{{
    html+='<div class="pair"><span class="pair-score">'+c.score.toFixed(4)+'</span> ';
    html+='<span class="pair-video">'+c.video+' @ frame '+c.position+'</span></div>';
  }});
  pairPanel.innerHTML=html;
}}

slider.addEventListener('input',()=>{{
  threshEl.textContent=parseFloat(slider.value).toFixed(3);
  drawHist();updatePairs();
}});

function saveThreshold(){{
  var thresh=parseFloat(slider.value);
  fetch('/api/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{threshold:thresh}})}}).then(r=>r.json()).then(d=>{{
    if(d.ok){{document.getElementById('savedMsg').style.display='inline';
      setTimeout(()=>document.getElementById('savedMsg').style.display='none',2000);}}
  }});
}}

window.addEventListener('resize',()=>{{drawHist();}});
drawHist();updatePairs();
threshEl.textContent=parseFloat(slider.value).toFixed(3);
</script></body></html>"""


def run_temporal_ui(
    embeddings_root: Path,
    images_root: Path | None = None,
    window_size: int = 120,
    min_gap: int = 30,
    exclude_folders: list[str] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    port: int = PORT,
    scores_cache: Path | None = None,
):
    """
    Launch the temporal threshold picker web UI.

    If scores_cache exists, loads from it instead of recomputing.
    """
    if images_root is None:
        images_root = _find_existing(_FRAMES_CANDIDATES, embeddings_root.parent / "Extracted Frames")

    # Load or compute scores
    if scores_cache and scores_cache.exists():
        print(f"[temporal_ui] Loading cached scores from {scores_cache}")
        with open(scores_cache) as f:
            data = json.load(f)
        all_scores = data["all_scores"]
        changes = data["changes"]
    else:
        print(f"[temporal_ui] Computing temporal scores (window={window_size})...")
        change_objs, scores_arr = compute_all_temporal_scores(
            embeddings_root, window_size, min_gap, exclude_folders)
        all_scores = scores_arr.tolist()
        changes = [
            {"video": c.video_name, "position": c.position,
             "score": c.change_score, "window_size": c.window_size}
            for c in change_objs
        ]
        # Cache for next time
        if scores_cache:
            scores_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_cache, "w") as f:
                json.dump({"all_scores": all_scores, "changes": changes}, f)
            print(f"[temporal_ui] Cached scores → {scores_cache}")

    html = _build_html(all_scores, changes, str(images_root), str(embeddings_root))
    saved_threshold = {"value": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

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
            if self.path == "/api/save":
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                d = json.loads(body)
                threshold = d["threshold"]
                saved_threshold["value"] = threshold
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump({"temporal_threshold": threshold, "window_size": window_size}, f, indent=2)
                print(f"[temporal_ui] Saved threshold={threshold:.4f} → {output_path}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
            else:
                self.send_error(404)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n[temporal_ui] Threshold picker at http://localhost:{port}")
    print(f"[temporal_ui] {len(all_scores)} scores, {len(changes)} local maxima")
    print("[temporal_ui] Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return saved_threshold["value"]


def load_temporal_threshold(path: Path = DEFAULT_OUTPUT) -> float | None:
    """Load saved threshold (for use in training without launching UI)."""
    if path.exists():
        with open(path) as f:
            return json.load(f).get("temporal_threshold")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal discontinuity threshold picker")
    default_emb = _find_existing(_EMBEDDINGS_CANDIDATES, _EMBEDDINGS_CANDIDATES[0])
    parser.add_argument("--embeddings-root", type=Path, default=default_emb)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--min-gap", type=int, default=30)
    parser.add_argument("--exclude", nargs="*", default=["reference for filtering"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--cache", type=Path,
                        default=PROJECT_ROOT / "output" / "temporal_scores_cache.json")
    args = parser.parse_args()

    run_temporal_ui(
        embeddings_root=args.embeddings_root,
        images_root=args.images_root,
        window_size=args.window_size,
        min_gap=args.min_gap,
        exclude_folders=args.exclude,
        output_path=args.output,
        port=args.port,
        scores_cache=args.cache,
    )


if __name__ == "__main__":
    main()
