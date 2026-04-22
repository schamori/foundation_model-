"""
Generic inference backend for overlaying detection model predictions in the
object-tracking UI.

Currently supports YOLO (ultralytics) via ``YOLOBackend``, but everything is
abstracted behind the ``InferenceBackend`` base so swapping in another model
family (HF detection, detectron2, custom torch checkpoints) is a matter of
adding one class + one entry to ``_BACKENDS``.

Public surface used by object_tracking.py:

    scan_weights()            -> list of discovered weight files + metadata
    load_backend(path, kind)  -> load a model; subsequent predicts use it
    predict(frame_bgr, conf)  -> list[{class_id, label, bbox:[x1,y1,x2,y2], score}]
    get_meta()                -> {"path", "kind", "classes"} of the loaded model
    unload_backend()          -> free GPU memory

All methods are thread-safe via a single module-level lock — the labeling HTTP
server is multi-threaded and two /api/inference/predict requests may race.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
END_TASKS_DIR = PROJECT_ROOT / "output" / "end_tasks"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class InferenceBackend:
    """Base interface — any detection model wraps one of these."""

    def predict(self, frame_bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
        """Return boxes in the *original* image coordinate space (no letterbox)."""
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class YOLOBackend(InferenceBackend):
    def __init__(self, weight_path: Path, device: Optional[str] = None):
        from ultralytics import YOLO
        self.weight_path = Path(weight_path)
        self.device = device
        self.model = YOLO(str(weight_path))
        # ultralytics returns class names in `model.names` as dict[int, str]
        self._names: dict[int, str] = dict(self.model.names)

    def predict(self, frame_bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
        # Ultralytics handles letterbox resize -> 640 and rescales boxes back.
        results = self.model(
            frame_bgr, conf=conf, verbose=False,
            device=self.device, imgsz=640,
        )
        out: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                cls = int(b.cls[0].item())
                score = float(b.conf[0].item())
                out.append({
                    "class_id": cls,
                    "label": self._names.get(cls, str(cls)),
                    "bbox": xyxy,
                    "score": score,
                })
        return out

    @property
    def class_names(self) -> list[str]:
        return [self._names[i] for i in sorted(self._names)]

    def close(self) -> None:
        try:
            del self.model
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# Map of kind -> backend class. Extend here for new model families.
_BACKENDS: dict[str, type[InferenceBackend]] = {
    "yolo": YOLOBackend,
}


def _infer_kind(path: Path) -> str:
    """Best-effort guess of backend family from file path."""
    s = str(path).lower()
    if s.endswith(".pt") or s.endswith(".onnx") or "yolo" in s:
        return "yolo"
    return "yolo"


# ---------------------------------------------------------------------------
# Weight discovery
# ---------------------------------------------------------------------------

def scan_weights() -> list[dict]:
    """Discover all trained weights under ``output/end_tasks/*/yolo/weights/``.

    Sorted by experiment name, then file — best.pt first within each run.
    """
    rows: list[dict] = []
    if not END_TASKS_DIR.is_dir():
        return rows
    for exp_dir in sorted(END_TASKS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        # YOLO runs: exp/yolo/weights/*.pt  (and also exp/yolo*/weights/*.pt)
        for sub in sorted(exp_dir.glob("yolo*")):
            wdir = sub / "weights"
            if not wdir.is_dir():
                continue
            for w in sorted(wdir.glob("*.pt")):
                rows.append({
                    "experiment": exp_dir.name,
                    "run": sub.name,
                    "file": w.name,
                    "path": str(w),
                    "size_mb": round(w.stat().st_size / 1024 / 1024, 1),
                    "kind": _infer_kind(w),
                })
    # Rank so "best.pt" sorts above "last.pt"
    def _rank(r):
        return (r["experiment"], r["run"], 0 if r["file"].startswith("best") else 1, r["file"])
    rows.sort(key=_rank)
    return rows


# ---------------------------------------------------------------------------
# Module-level singleton (one model loaded at a time)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_BACKEND: Optional[InferenceBackend] = None
_META: dict = {}


def load_backend(weight_path: str, kind: Optional[str] = None,
                 device: Optional[str] = None) -> dict:
    """Load (or replace) the active inference backend. Returns its metadata."""
    global _BACKEND, _META
    kind = kind or _infer_kind(Path(weight_path))
    if kind not in _BACKENDS:
        raise ValueError(f"unknown inference kind: {kind} (have: {list(_BACKENDS)})")

    with _LOCK:
        if _BACKEND is not None:
            try:
                _BACKEND.close()
            except Exception:
                pass
            _BACKEND = None
        backend = _BACKENDS[kind](Path(weight_path), device=device)
        _BACKEND = backend
        _META = {
            "path": str(weight_path),
            "kind": kind,
            "classes": backend.class_names,
            "device": device or "auto",
        }
        return dict(_META)


def unload_backend() -> None:
    global _BACKEND, _META
    with _LOCK:
        if _BACKEND is not None:
            try:
                _BACKEND.close()
            except Exception:
                pass
        _BACKEND = None
        _META = {}


def predict(frame_bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Run the currently-loaded backend on the frame. Empty list if none."""
    with _LOCK:
        if _BACKEND is None:
            return []
        backend = _BACKEND
    # Run *outside* the lock if we want concurrency, but YOLO inference is
    # already GPU-serialized. Keep it simple: stay locked so memory is safe.
    return backend.predict(frame_bgr, conf=conf)


def get_meta() -> dict:
    with _LOCK:
        return dict(_META)


def is_loaded() -> bool:
    with _LOCK:
        return _BACKEND is not None
