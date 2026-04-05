"""View and delete anti-reference images. Opens a local web page."""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import webbrowser

ANTI_REF_DIR = Path("/media/HDD1/moritz/Extracted Frames/anti_references")
FEATURES_ROOT = Path("/media/HDD1/moritz/Extracted Frames embeddings")
IMAGES_ROOT = Path("/media/HDD1/moritz/Extracted Frames")
PORT = 8769


def feature_path_to_image_path(feature_path: Path) -> Path | None:
    relative = feature_path.relative_to(FEATURES_ROOT)
    stem_path = IMAGES_ROOT / relative.with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".PNG"):
        candidate = stem_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_all():
    """Load all anti-ref files, return {filename: [{"npy": str, "img": str|None}, ...]}"""
    result = {}
    if not ANTI_REF_DIR.exists():
        return result
    for f in sorted(ANTI_REF_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        entries = []
        for p in data.get("paths", []):
            pp = Path(p)
            img = feature_path_to_image_path(pp) if pp.exists() else None
            entries.append({"npy": p, "img": str(img) if img else None, "exists": pp.exists()})
        result[f.name] = entries
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._serve_page()
        elif self.path == "/api/data":
            self._json(load_all())
        elif self.path.startswith("/img?"):
            qs = parse_qs(self.path.split("?", 1)[1])
            p = qs.get("p", [""])[0]
            if p and Path(p).exists():
                data = Path(p).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fname = body["file"]
            npy_paths = set(body["paths"])
            fpath = ANTI_REF_DIR / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                before = len(data["paths"])
                data["paths"] = [p for p in data["paths"] if p not in npy_paths]
                with open(fpath, "w") as f:
                    json.dump(data, f)
                removed = before - len(data["paths"])
                self._json({"ok": True, "removed": removed, "remaining": len(data["paths"])})
            else:
                self._json({"ok": False, "error": "file not found"})
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def _serve_page(self):
        html = PAGE_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)


PAGE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Anti-References Viewer</title>
<style>
body { font-family: system-ui; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
h1 { color: #e94560; }
h2 { color: #0f3460; background: #e94560; display: inline-block; padding: 4px 12px; border-radius: 4px; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 30px; }
.card { background: #16213e; border-radius: 8px; padding: 8px; width: 280px; position: relative; }
.card img { width: 100%; border-radius: 4px; }
.card .path { font-size: 11px; color: #888; word-break: break-all; margin-top: 4px; }
.card .missing { color: #e94560; font-weight: bold; }
.card .cb { position: absolute; top: 12px; right: 12px; width: 24px; height: 24px; accent-color: #e94560; }
.btn { background: #e94560; color: #fff; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 16px; margin: 10px 4px; }
.btn:hover { background: #c73650; }
.btn:disabled { opacity: 0.4; }
.status { color: #0f3460; font-weight: bold; margin: 10px 0; }
</style></head><body>
<h1>Anti-References Viewer</h1>
<div id="controls">
  <button class="btn" onclick="selectAll()">Select All</button>
  <button class="btn" onclick="selectNone()">Select None</button>
  <button class="btn" id="delBtn" onclick="deleteSelected()" disabled>Delete Selected (0)</button>
</div>
<div id="status"></div>
<div id="content">Loading...</div>
<script>
let allData = {};

async function load() {
  const r = await fetch('/api/data');
  allData = await r.json();
  render();
}

function render() {
  const c = document.getElementById('content');
  let html = '';
  for (const [fname, entries] of Object.entries(allData)) {
    html += `<h2>${fname} (${entries.length})</h2><div class="grid">`;
    entries.forEach((e, i) => {
      const id = `${fname}::${i}`;
      html += `<div class="card">`;
      html += `<input type="checkbox" class="cb" data-file="${fname}" data-npy="${e.npy}" onchange="updateBtn()">`;
      if (e.img) {
        html += `<img src="/img?p=${encodeURIComponent(e.img)}" loading="lazy">`;
      } else if (!e.exists) {
        html += `<div class="missing" style="height:180px;display:flex;align-items:center;justify-content:center">Embedding deleted</div>`;
      } else {
        html += `<div class="missing" style="height:180px;display:flex;align-items:center;justify-content:center">No image found</div>`;
      }
      html += `<div class="path">${e.npy.split('/').slice(-2).join('/')}</div>`;
      html += `</div>`;
    });
    html += `</div>`;
  }
  if (!html) html = '<p>No anti-reference files found.</p>';
  c.innerHTML = html;
}

function getChecked() {
  return [...document.querySelectorAll('.cb:checked')];
}

function updateBtn() {
  const n = getChecked().length;
  const btn = document.getElementById('delBtn');
  btn.textContent = `Delete Selected (${n})`;
  btn.disabled = n === 0;
}

function selectAll() { document.querySelectorAll('.cb').forEach(c => c.checked = true); updateBtn(); }
function selectNone() { document.querySelectorAll('.cb').forEach(c => c.checked = false); updateBtn(); }

async function deleteSelected() {
  const checked = getChecked();
  if (!checked.length) return;
  if (!confirm(`Remove ${checked.length} anti-reference(s)?`)) return;

  // Group by file
  const byFile = {};
  checked.forEach(cb => {
    const f = cb.dataset.file;
    if (!byFile[f]) byFile[f] = [];
    byFile[f].push(cb.dataset.npy);
  });

  const status = document.getElementById('status');
  for (const [fname, paths] of Object.entries(byFile)) {
    const r = await fetch('/api/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file: fname, paths})
    });
    const d = await r.json();
    status.textContent = `${fname}: removed ${d.removed}, ${d.remaining} remaining`;
  }
  load();
}

load();
</script></body></html>
"""

if __name__ == "__main__":
    print(f"Anti-references viewer at http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
