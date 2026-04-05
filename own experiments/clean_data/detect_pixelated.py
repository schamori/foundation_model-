"""
Detect pixelated / mosaic-block images by estimated block size.

Detects periodic grid pattern in gradients via autocorrelation.
Shows images with block size >= threshold, sorted largest first.

Usage:
    python detect_pixelated.py [/path/to/image_folder]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_DIM = 800
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _autocorr_peak(signal: np.ndarray) -> tuple[float, int]:
    n = len(signal)
    if n < 16:
        return 0.0, 0
    sig = signal - np.mean(signal)
    norm = np.sum(sig ** 2)
    if norm < 1e-10:
        return 0.0, 0
    max_lag = min(n // 3, 80)
    min_lag = 3
    if max_lag <= min_lag:
        return 0.0, 0
    acorr = np.array([
        np.sum(sig[:n - lag] * sig[lag:]) / norm
        for lag in range(min_lag, max_lag)
    ])
    if len(acorr) == 0:
        return 0.0, 0
    peak_idx = np.argmax(acorr)
    return float(acorr[peak_idx]), int(peak_idx + min_lag)


def detect_block_size(gray: np.ndarray) -> int:
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0
    n_samples = min(50, h, w)
    row_indices = np.linspace(0, h - 1, n_samples, dtype=int)
    col_indices = np.linspace(0, w - 1, n_samples, dtype=int)
    all_block_sizes = []
    for idx in row_indices:
        grad = np.abs(np.diff(gray[idx, :].astype(np.float64)))
        _, bs = _autocorr_peak(grad)
        if bs > 0:
            all_block_sizes.append(bs)
    for idx in col_indices:
        grad = np.abs(np.diff(gray[:, idx].astype(np.float64)))
        _, bs = _autocorr_peak(grad)
        if bs > 0:
            all_block_sizes.append(bs)
    if not all_block_sizes:
        return 0
    counts = np.bincount(np.array(all_block_sizes))
    return int(np.argmax(counts))


def _score_one(image_path: str) -> dict | None:
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < 32 or h < 32:
            return None
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
        gray = np.array(img.convert("L"))
        bs = detect_block_size(gray)
        return {"path": image_path, "bs": bs}
    except Exception:
        return None


def scan_paths(paths: list[Path], workers: int = 8) -> list[dict]:
    from multiprocessing import Pool
    from tqdm import tqdm
    str_paths = [str(p) for p in paths]
    print(f"Scanning {len(str_paths)} images with {workers} workers...")
    results = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap_unordered(_score_one, str_paths, chunksize=64),
                      total=len(str_paths), desc="Scoring"):
            if r is not None:
                results.append(r)
    return results


def serve_viewer(results: list[dict], port: int = 8769):
    import json as _json
    import urllib.parse
    from http.server import HTTPServer, BaseHTTPRequestHandler

    results.sort(key=lambda r: r["bs"], reverse=True)

    all_items = _json.dumps([
        {"p": r["path"], "bs": r["bs"]} for r in results
    ])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Block size detector</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column}}
  #bar{{padding:10px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}}
  #counter{{font-weight:bold;font-size:1.1em}}
  #mBs{{font-size:1em;color:#f90}}
  #name{{color:#666;font-size:.8em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:40%}}
  #hint{{font-size:.8em;color:#555;margin-left:auto}}
  #thresh-bar{{padding:6px 16px;background:#222;display:flex;align-items:center;gap:12px;flex-shrink:0}}
  #thresh-bar label{{font-size:.85em;color:#aaa}}
  #thresh-bar input[type=number]{{width:60px;background:#333;color:#eee;border:1px solid #555;padding:2px 6px;font-size:.9em}}
  #stats{{font-size:.85em;color:#888;margin-left:16px}}
  #imgwrap{{flex:1;display:flex;align-items:center;justify-content:center;padding:8px;min-height:0}}
  #imgwrap img{{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}}
  .flagged{{outline:3px solid #f44;border-radius:4px}}
</style></head><body>
<div id="bar">
  <span id="counter"></span>
  <span id="mBs"></span>
  <span id="name"></span>
  <span id="hint">a/d navigate &nbsp; Enter = delete all shown &nbsp; Ctrl+C done</span>
</div>
<div id="thresh-bar">
  <label>Min block size:</label>
  <input type="number" id="bsInput" value="17" min="0" max="80">
  <span id="stats"></span>
</div>
<div id="imgwrap"><img id="img"></div>
<script>
var allItems={all_items};
var minBs=17;
var filtered=[];
var i=0;
var bsInput=document.getElementById('bsInput');
bsInput.addEventListener('input',function(){{minBs=parseInt(bsInput.value)||0;refilter();i=0;show();}});
function refilter(){{
  filtered=allItems.filter(x=>x.bs>=minBs);
  document.getElementById('stats').textContent=filtered.length+' / '+allItems.length+' images with block size >= '+minBs;
}}
function show(){{
  if(filtered.length===0){{
    document.getElementById('counter').textContent='0/0';
    document.getElementById('mBs').textContent='';
    document.getElementById('name').textContent='No images match';
    document.getElementById('img').src='';
    document.getElementById('img').className='';
    return;
  }}
  if(i>=filtered.length)i=filtered.length-1;
  if(i<0)i=0;
  var it=filtered[i];
  document.getElementById('counter').textContent=(i+1)+'/'+filtered.length;
  document.getElementById('mBs').textContent='block = '+it.bs+'px';
  document.getElementById('name').textContent=it.p.split('/').slice(-2).join('/');
  document.getElementById('img').src='/img?p='+encodeURIComponent(it.p)+'&i='+i;
  document.getElementById('img').className='flagged';
}}
document.addEventListener('keydown',function(e){{
  if(e.key==='d'||e.key==='ArrowRight'){{if(i<filtered.length-1){{i++;show();}}}}
  else if(e.key==='a'||e.key==='ArrowLeft'){{if(i>0){{i--;show();}}}}
  else if(e.key==='Enter'){{
    if(filtered.length>0&&confirm('Delete '+filtered.length+' images with block size >= '+minBs+'?')){{
      fetch('/delete',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{min_bs:minBs}})}})
      .then(r=>r.json()).then(d=>{{alert('Deleted '+d.deleted+' images');location.reload();}});
    }}
  }}
}});
refilter();
show();
</script>
</body></html>"""

    deleted_count = [0]

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
            if self.path == "/delete":
                data = _json.loads(body)
                mb = int(data["min_bs"])
                count = 0
                for r in results:
                    if r["bs"] >= mb:
                        p = Path(r["path"])
                        if p.exists():
                            p.unlink()
                            count += 1
                deleted_count[0] += count
                print(f"[pixelated] Deleted {count} images with block size >= {mb}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json.dumps({"deleted": count}).encode())
            else:
                self.send_error(404)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nViewer at  http://localhost:{port}")
    print(f"SSH tunnel: ssh -L {port}:localhost:{port} user@host")
    print("Showing images with block size >= threshold (largest first).")
    print("Enter to delete, Ctrl+C when done.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print(f"Total deleted this session: {deleted_count[0]}")


def main():
    images_root = Path(r"/media/HDD1/moritz/Extracted Frames")
    exclude_folders = {"reference for filtering", "reference images", "anti_references"}

    target = Path(sys.argv[1]) if len(sys.argv) >= 2 else images_root
    if not target.is_dir():
        print(f"Not a directory: {target}")
        sys.exit(1)

    paths = sorted(
        p for p in target.rglob("*")
        if p.suffix.lower() in _IMG_EXTS
        and not any(ex in p.parts for ex in exclude_folders)
    )
    results = scan_paths(paths)
    print(f"Scanned {len(results)} images")
    serve_viewer(results)


if __name__ == "__main__":
    main()
