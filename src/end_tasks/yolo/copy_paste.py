"""
Copy-paste augmentation for instrument detection — rare-class booster.

Idea: scan the training set, crop every instance (per class), and at train time
occasionally paste a rare-class instance (e.g. CUSA, Microscissors) onto an
image to improve class balance. Cheap and effective when a handful of classes
are under-represented.

Design
------
- ``InstanceBank``: one-time scan of a YOLO-format dataset. Crops every bbox
  above a size threshold and stores it in memory keyed by class id.
- ``CopyPasteAugmentor``: callable ``(img, bboxes) -> (img, bboxes)``. Applies
  pastes with probability ``p``, picks source instances from the bank, rejects
  placements that overlap too much with existing boxes, optionally feathers
  the paste edges for a softer blend.

NOT yet wired into the ultralytics training loop — to integrate you'd either
register it as a ultralytics callback on the train dataloader's __getitem__,
or pre-augment images offline. See ``__main__`` below for an interactive
preview server that lets you tune parameters visually.

Run the preview:
    python -m src.end_tasks.yolo.copy_paste \\
        --dataset-dir output/end_tasks/instruments_yolov12/yolo_dataset \\
        --port 8766
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..base import INSTRUMENTS

# Rare classes by default — index positions in INSTRUMENTS
DEFAULT_RARE_CLASSES: tuple[int, ...] = (
    INSTRUMENTS.index("CUSA"),
    INSTRUMENTS.index("Microscissors"),
)


# ---------------------------------------------------------------------------
# Instance bank
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    class_id: int
    crop: np.ndarray  # HxWx3, BGR, uint8
    src: str


@dataclass
class InstanceBank:
    """Scans a YOLO dataset and caches cropped instances per class."""

    dataset_dir: Path
    split: str = "train"
    min_size: int = 24              # reject tiny crops (px, min of W/H)
    max_size: int = 512             # reject huge crops (px, max of W/H)
    class_filter: Optional[tuple[int, ...]] = None

    instances: list[Instance] = field(default_factory=list)
    by_class: dict[int, list[int]] = field(default_factory=dict)  # cls → indices

    def load(self) -> "InstanceBank":
        labels_dir = self.dataset_dir / "labels" / self.split
        images_dir = self.dataset_dir / "images" / self.split
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"labels dir missing: {labels_dir}")

        for lbl_path in sorted(labels_dir.glob("*.txt")):
            img_path = _find_image(images_dir, lbl_path.stem)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for cls, cx, cy, bw, bh in _parse_yolo_lines(lbl_path):
                if self.class_filter is not None and cls not in self.class_filter:
                    continue
                x1 = int(round((cx - bw / 2) * w))
                y1 = int(round((cy - bh / 2) * h))
                x2 = int(round((cx + bw / 2) * w))
                y2 = int(round((cy + bh / 2) * h))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cw, ch = x2 - x1, y2 - y1
                if min(cw, ch) < self.min_size or max(cw, ch) > self.max_size:
                    continue
                crop = img[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                idx = len(self.instances)
                self.instances.append(Instance(class_id=cls, crop=crop, src=img_path.name))
                self.by_class.setdefault(cls, []).append(idx)
        return self

    def class_counts(self) -> dict[int, int]:
        return {c: len(ix) for c, ix in self.by_class.items()}

    def sample(self, class_ids: Optional[list[int]], rng: random.Random) -> Optional[Instance]:
        pool: list[int] = []
        cls_set = set(class_ids) if class_ids is not None else None
        for cls, ixs in self.by_class.items():
            if cls_set is not None and cls not in cls_set:
                continue
            pool.extend(ixs)
        if not pool:
            return None
        return self.instances[rng.choice(pool)]


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists() or p.is_symlink():
            return p
    return None


def _parse_yolo_lines(path: Path):
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        yield cls, cx, cy, bw, bh


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------

@dataclass
class CopyPasteAugmentor:
    """Pastes rare-class instances onto images.

    All bboxes are YOLO-normalized: (class, cx, cy, w, h).
    Returns a new (img, bboxes) — does not mutate inputs.
    """

    bank: InstanceBank
    p: float = 0.3                              # probability the aug fires per image
    max_pastes: int = 2                         # up to N pastes when it fires
    target_classes: Optional[list[int]] = None  # None → all classes in bank
    scale_range: tuple[float, float] = (0.5, 1.2)
    rotation_deg: float = 0.0                   # ±deg uniform; 0 disables
    max_iou: float = 0.1                        # skip placements with IoU > this
    feather: int = 3                            # Gaussian blur radius (0 = hard paste)
    placement_tries: int = 20

    def __call__(
        self,
        img: np.ndarray,
        bboxes: list[tuple[int, float, float, float, float]],
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        rng = random.Random(seed)
        if rng.random() > self.p or not self.bank.instances:
            return img, list(bboxes)

        h, w = img.shape[:2]
        out = img.copy()
        out_boxes = list(bboxes)
        n_paste = rng.randint(1, max(1, self.max_pastes))

        for _ in range(n_paste):
            inst = self.bank.sample(self.target_classes, rng)
            if inst is None:
                break
            pasted = _prepare_paste(inst.crop, self.scale_range, self.rotation_deg, rng)
            if pasted is None:
                continue
            ph, pw = pasted["img"].shape[:2]
            if ph >= h or pw >= w:
                continue

            placed = False
            for _ in range(self.placement_tries):
                px = rng.randint(0, w - pw)
                py = rng.randint(0, h - ph)
                new_box = (
                    inst.class_id,
                    (px + pw / 2) / w,
                    (py + ph / 2) / h,
                    pw / w,
                    ph / h,
                )
                if _max_iou(new_box[1:], out_boxes) <= self.max_iou:
                    _blit(out, pasted["img"], pasted["alpha"], px, py, self.feather)
                    out_boxes.append(new_box)
                    placed = True
                    break
            if not placed:
                continue
        return out, out_boxes


def _prepare_paste(
    crop: np.ndarray,
    scale_range: tuple[float, float],
    rotation_deg: float,
    rng: random.Random,
) -> Optional[dict]:
    ch, cw = crop.shape[:2]
    scale = rng.uniform(*scale_range)
    new_w = max(8, int(cw * scale))
    new_h = max(8, int(ch * scale))
    scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    alpha = np.full((new_h, new_w), 255, dtype=np.uint8)

    if rotation_deg > 0:
        ang = rng.uniform(-rotation_deg, rotation_deg)
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        out_w = int(new_h * sin + new_w * cos)
        out_h = int(new_h * cos + new_w * sin)
        M[0, 2] += out_w / 2 - new_w / 2
        M[1, 2] += out_h / 2 - new_h / 2
        scaled = cv2.warpAffine(scaled, M, (out_w, out_h), borderValue=(0, 0, 0))
        alpha = cv2.warpAffine(alpha, M, (out_w, out_h), borderValue=0)

    # Tighten to non-zero alpha bounding box to drop rotation black borders
    ys, xs = np.where(alpha > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    scaled = scaled[y0:y1, x0:x1]
    alpha = alpha[y0:y1, x0:x1]
    return {"img": scaled, "alpha": alpha}


def _blit(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, x: int, y: int, feather: int) -> None:
    ph, pw = src.shape[:2]
    if feather > 0:
        k = 2 * feather + 1
        a = cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0) / 255.0
    else:
        a = alpha.astype(np.float32) / 255.0
    a = a[:, :, None]
    roi = dst[y:y + ph, x:x + pw].astype(np.float32)
    blend = src.astype(np.float32) * a + roi * (1.0 - a)
    dst[y:y + ph, x:x + pw] = np.clip(blend, 0, 255).astype(np.uint8)


def _iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x1a, y1a, x2a, y2a = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x1b, y1b, x2b, y2b = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1e-9)


def _max_iou(box, existing) -> float:
    if not existing:
        return 0.0
    return max(_iou(box, (e[1], e[2], e[3], e[4])) for e in existing)


# ---------------------------------------------------------------------------
# Preview server
# ---------------------------------------------------------------------------

_CLASS_COLORS = [
    (255, 99, 132), (54, 162, 235), (255, 205, 86), (75, 192, 192),
    (153, 102, 255), (255, 159, 64), (180, 180, 180),
]


def _draw_boxes(img: np.ndarray, boxes, pasted_count: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    n = len(boxes)
    for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
        color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
        color = (color[2], color[1], color[0])  # RGB → BGR
        is_pasted = i >= (n - pasted_count)
        thickness = 3 if is_pasted else 2
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{INSTRUMENTS[cls]}{' *' if is_pasted else ''}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def _load_sample(dataset_dir: Path, split: str, idx: int):
    labels_dir = dataset_dir / "labels" / split
    images_dir = dataset_dir / "images" / split
    all_labels = sorted(labels_dir.glob("*.txt"))
    if not all_labels:
        raise FileNotFoundError(f"no labels in {labels_dir}")
    idx = idx % len(all_labels)
    lbl = all_labels[idx]
    img_path = _find_image(images_dir, lbl.stem)
    if img_path is None:
        raise FileNotFoundError(f"no image for {lbl.stem}")
    img = cv2.imread(str(img_path))
    boxes = list(_parse_yolo_lines(lbl))
    return img, boxes, lbl.stem, len(all_labels)


class _Handler(BaseHTTPRequestHandler):
    dataset_dir: Path = None
    split: str = "train"
    bank: InstanceBank = None

    def log_message(self, fmt, *args):  # silence default access logs
        return

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", _HTML.encode())
            elif path == "/api/info":
                self._api_info()
            elif path == "/api/preview":
                self._api_preview(query)
            else:
                self.send_error(404)
        except Exception as e:
            self.send_error(500, f"{type(e).__name__}: {e}")

    # -- handlers -----------------------------------------------------------
    def _api_info(self):
        counts = self.bank.class_counts()
        n_samples = len(list((self.dataset_dir / "labels" / self.split).glob("*.txt")))
        body = {
            "instruments": INSTRUMENTS,
            "class_counts": {int(k): v for k, v in counts.items()},
            "n_samples": n_samples,
            "default_rare": list(DEFAULT_RARE_CLASSES),
        }
        self._send(200, "application/json", json.dumps(body).encode())

    def _api_preview(self, q):
        idx = int(q.get("idx", ["0"])[0])
        seed = int(q.get("seed", ["0"])[0])
        p = float(q.get("p", ["1.0"])[0])
        max_pastes = int(q.get("max_pastes", ["2"])[0])
        scale_min = float(q.get("scale_min", ["0.5"])[0])
        scale_max = float(q.get("scale_max", ["1.2"])[0])
        rotation = float(q.get("rotation", ["0"])[0])
        feather = int(q.get("feather", ["3"])[0])
        max_iou_ = float(q.get("max_iou", ["0.1"])[0])
        cls_str = q.get("classes", [",".join(str(c) for c in DEFAULT_RARE_CLASSES)])[0]
        classes = [int(x) for x in cls_str.split(",") if x.strip() != ""]

        img, boxes, stem, total = _load_sample(self.dataset_dir, self.split, idx)

        aug = CopyPasteAugmentor(
            bank=self.bank,
            p=p,
            max_pastes=max_pastes,
            target_classes=classes if classes else None,
            scale_range=(min(scale_min, scale_max), max(scale_min, scale_max)),
            rotation_deg=rotation,
            max_iou=max_iou_,
            feather=feather,
        )
        n_before = len(boxes)
        out_img, out_boxes = aug(img, boxes, seed=seed)
        pasted_count = len(out_boxes) - n_before

        drawn = _draw_boxes(out_img, out_boxes, pasted_count)
        ok, buf = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("X-Sample-Stem", stem)
        self.send_header("X-Sample-Total", str(total))
        self.send_header("X-Pasted-Count", str(pasted_count))
        self.send_header("X-Original-Count", str(n_before))
        self.end_headers()
        self.wfile.write(buf.tobytes())

    # -- utils --------------------------------------------------------------
    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# HTML/JS
# ---------------------------------------------------------------------------

_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Copy-Paste Preview</title>
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; display: flex; height: 100vh; }
#sidebar { width: 340px; min-width: 340px; padding: 16px; border-right: 1px solid #333; overflow-y: auto; }
#main { flex: 1; display: flex; flex-direction: column; padding: 12px; }
#img-wrap { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; border: 1px solid #222; min-height: 0; }
#img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
.row { margin: 10px 0; }
.row label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 3px; }
.row input[type=range] { width: 100%; }
.row input[type=number] { width: 70px; background: #222; color: #eee; border: 1px solid #444; padding: 3px 5px; }
h1 { font-size: 16px; margin: 0 0 12px; color: #ffa; }
.cls-list { display: grid; grid-template-columns: 1fr 1fr; gap: 3px; font-size: 12px; }
.cls-list label { cursor: pointer; padding: 3px 5px; background: #222; border-radius: 3px; }
.cls-list label.active { background: #2a5; color: #fff; }
.cls-list label .cnt { opacity: 0.6; margin-left: 4px; font-size: 10px; }
.meta { font-size: 11px; color: #888; margin-top: 6px; font-family: monospace; }
.btn-row { display: flex; gap: 4px; margin: 6px 0; }
button { background: #333; color: #eee; border: 1px solid #555; padding: 6px 10px; border-radius: 3px; cursor: pointer; }
button:hover { background: #444; }
#status { font-size: 11px; color: #aaa; font-family: monospace; padding: 4px 0; }
</style>
</head><body>
<div id="sidebar">
  <h1>Copy-Paste Augmentor</h1>

  <div class="btn-row">
    <button id="prev">← Prev</button>
    <input id="idx" type="number" value="0" min="0" style="flex:1"/>
    <button id="next">Next →</button>
  </div>
  <div class="btn-row">
    <button id="rand">Random</button>
    <button id="reseed">New seed</button>
  </div>

  <div class="row">
    <label>Probability (p) <span id="p-v">1.0</span></label>
    <input type="range" id="p" min="0" max="1" step="0.05" value="1.0">
  </div>
  <div class="row">
    <label>Max pastes <span id="max_pastes-v">2</span></label>
    <input type="range" id="max_pastes" min="1" max="8" step="1" value="2">
  </div>
  <div class="row">
    <label>Scale min <span id="scale_min-v">0.5</span></label>
    <input type="range" id="scale_min" min="0.1" max="2.0" step="0.05" value="0.5">
  </div>
  <div class="row">
    <label>Scale max <span id="scale_max-v">1.2</span></label>
    <input type="range" id="scale_max" min="0.1" max="2.0" step="0.05" value="1.2">
  </div>
  <div class="row">
    <label>Rotation ±° <span id="rotation-v">0</span></label>
    <input type="range" id="rotation" min="0" max="45" step="1" value="0">
  </div>
  <div class="row">
    <label>Feather (blend) <span id="feather-v">3</span></label>
    <input type="range" id="feather" min="0" max="30" step="1" value="3">
  </div>
  <div class="row">
    <label>Max IoU overlap <span id="max_iou-v">0.10</span></label>
    <input type="range" id="max_iou" min="0" max="0.9" step="0.05" value="0.10">
  </div>
  <div class="row">
    <label>Seed <span></span></label>
    <input type="number" id="seed" value="0" style="width:100%"/>
  </div>

  <div class="row">
    <label>Paste classes</label>
    <div id="cls-list" class="cls-list"></div>
  </div>

  <div id="status"></div>
</div>

<div id="main">
  <div id="img-wrap"><img id="img" alt="preview"/></div>
  <div class="meta" id="meta">loading...</div>
</div>

<script>
const params = ["p","max_pastes","scale_min","scale_max","rotation","feather","max_iou","seed","idx"];
const state = {
  p: 1.0, max_pastes: 2, scale_min: 0.5, scale_max: 1.2,
  rotation: 0, feather: 3, max_iou: 0.10, seed: 0, idx: 0,
  classes: new Set(),
  total: 0,
};

function bindSlider(id) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "-v");
  el.addEventListener("input", () => {
    state[id] = Number(el.value);
    if (out) out.textContent = el.value;
    refresh();
  });
}

async function loadInfo() {
  const res = await fetch("/api/info");
  const info = await res.json();
  state.total = info.n_samples;
  document.getElementById("idx").max = info.n_samples - 1;

  // default rare classes selected
  info.default_rare.forEach(c => state.classes.add(c));

  const list = document.getElementById("cls-list");
  list.innerHTML = "";
  info.instruments.forEach((name, i) => {
    const count = info.class_counts[i] || 0;
    const label = document.createElement("label");
    label.innerHTML = `${name}<span class="cnt">(${count})</span>`;
    if (state.classes.has(i)) label.classList.add("active");
    if (count === 0) label.style.opacity = 0.3;
    label.addEventListener("click", () => {
      if (state.classes.has(i)) { state.classes.delete(i); label.classList.remove("active"); }
      else { state.classes.add(i); label.classList.add("active"); }
      refresh();
    });
    list.appendChild(label);
  });
}

function buildUrl() {
  const qs = new URLSearchParams();
  for (const k of params) qs.set(k, state[k]);
  qs.set("classes", [...state.classes].join(","));
  return "/api/preview?" + qs.toString();
}

async function refresh() {
  const url = buildUrl();
  const res = await fetch(url);
  if (!res.ok) { document.getElementById("status").textContent = "error: " + res.status; return; }
  const blob = await res.blob();
  document.getElementById("img").src = URL.createObjectURL(blob);
  const stem = res.headers.get("X-Sample-Stem");
  const total = res.headers.get("X-Sample-Total");
  const pasted = res.headers.get("X-Pasted-Count");
  const orig = res.headers.get("X-Original-Count");
  document.getElementById("meta").textContent =
    `[${state.idx}/${total}] ${stem}  —  original boxes: ${orig}, pasted: ${pasted}`;
}

document.getElementById("prev").onclick = () => { state.idx = (state.idx - 1 + state.total) % state.total; document.getElementById("idx").value = state.idx; refresh(); };
document.getElementById("next").onclick = () => { state.idx = (state.idx + 1) % state.total; document.getElementById("idx").value = state.idx; refresh(); };
document.getElementById("rand").onclick = () => { state.idx = Math.floor(Math.random() * state.total); document.getElementById("idx").value = state.idx; refresh(); };
document.getElementById("reseed").onclick = () => { state.seed = Math.floor(Math.random() * 1e9); document.getElementById("seed").value = state.seed; refresh(); };
document.getElementById("idx").addEventListener("change", (e) => { state.idx = Number(e.target.value); refresh(); });
document.getElementById("seed").addEventListener("change", (e) => { state.seed = Number(e.target.value); refresh(); });

params.filter(p => p !== "seed" && p !== "idx").forEach(bindSlider);

loadInfo().then(refresh);
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    from ..config import PROJECT_ROOT
    return PROJECT_ROOT / "output" / "end_tasks" / "instruments_yolov12" / "yolo_dataset"


def main():
    parser = argparse.ArgumentParser(description="Copy-paste augmentation preview server")
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir(),
                        help="YOLO dataset root (with labels/{split}/ and images/{split}/)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--min-size", type=int, default=24)
    parser.add_argument("--max-size", type=int, default=512)
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise SystemExit(
            f"[copy-paste] dataset dir missing: {args.dataset_dir}\n"
            f"Run training once first (or `python -m src.end_tasks.yolo.export_only`)."
        )

    print(f"[copy-paste] scanning {args.dataset_dir / 'labels' / args.split}...")
    bank = InstanceBank(
        dataset_dir=args.dataset_dir,
        split=args.split,
        min_size=args.min_size,
        max_size=args.max_size,
    ).load()
    counts = bank.class_counts()
    print(f"[copy-paste] bank loaded: {len(bank.instances)} crops")
    for c, n in sorted(counts.items()):
        marker = " (rare)" if c in DEFAULT_RARE_CLASSES else ""
        print(f"  {INSTRUMENTS[c]:<20} cls={c}  n={n}{marker}")

    _Handler.dataset_dir = args.dataset_dir
    _Handler.split = args.split
    _Handler.bank = bank

    httpd = HTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n[copy-paste] preview server ready — open {url}\n  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[copy-paste] shutting down.")


if __name__ == "__main__":
    main()
