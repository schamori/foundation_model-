#!/usr/bin/env python3
"""
Surgical Instrument Video Tracker — SAM2 + Editable Bounding Boxes

Features:
  - SAM2 box prompts for each instrument
  - Masks → bboxes via largest N connected components (configurable blobs)
  - Show both masks AND bounding boxes (separate toggles)
  - Editable bboxes (drag move/resize) per frame
  - Change object class/label after tracking
  - Mark object "out of frame" per frame
  - Delete entire object series
  - Autosave/autoload — resume where you left off
  - Multiple tracking runs merge (no reset needed)
  - Always-interactive (no mode lock)

Install SAM2:
  pip install sam2   # or:
  git clone https://github.com/facebookresearch/sam2.git ~/sam2
  cd ~/sam2 && pip install -e .
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os, sys, json, shutil, subprocess, tempfile, threading, traceback, time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
import torch

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
VIDEO_PATH    = Path("/media/HDD1/moritz/foundential/Anterior Temporal Lobe Resection Operative Videos/ATLR_20.mp4")
PORT          = 8765
SAMPLE_FPS    = 1
TRACK_SECONDS = 50
SAM2_HF_ID   = "facebook/sam2-hiera-large"
EXPORT_DIR    = Path("./tracking_exports")
AUTOSAVE_DIR  = EXPORT_DIR / "autosave" / VIDEO_PATH.stem

INSTRUMENTS = [
    {"name": "Bipolar",         "hex": "#ef476f", "rgb": [239,  71, 111]},
    {"name": "Microdissectors", "hex": "#06d6a0", "rgb": [  6, 214, 160]},
    {"name": "Suction",         "hex": "#118ab2", "rgb": [ 17, 138, 178]},
    {"name": "Drills",          "hex": "#ffd166", "rgb": [255, 209, 102]},
    {"name": "Microscissors",   "hex": "#9b59b6", "rgb": [155,  89, 182]},
    {"name": "CUSA",            "hex": "#e67e22", "rgb": [230, 126,  34]},
    {"name": "Others",          "hex": "#9aa5b1", "rgb": [154, 165, 177]},
]
INST_COLOR = {i["name"]: tuple(i["rgb"]) for i in INSTRUMENTS}

# ═══════════════════════════════════════════════════════════════
#  MASK → BOUNDING BOX  (N largest connected components)
# ═══════════════════════════════════════════════════════════════
def mask_to_bbox(mask, n_blobs=1):
    """Find n largest connected components, return ONE encompassing bbox."""
    if mask is None or not np.any(mask):
        return None
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(-areas) + 1
    n_take = min(n_blobs, len(areas))
    x1, y1, x2, y2 = float('inf'), float('inf'), 0, 0
    for idx in sorted_idx[:n_take]:
        bx = stats[idx, cv2.CC_STAT_LEFT]
        by = stats[idx, cv2.CC_STAT_TOP]
        bw = stats[idx, cv2.CC_STAT_WIDTH]
        bh = stats[idx, cv2.CC_STAT_HEIGHT]
        x1, y1 = min(x1, bx), min(y1, by)
        x2, y2 = max(x2, bx + bw), max(y2, by + bh)
    return [int(x1), int(y1), int(x2), int(y2)]

def mask_keep_n_blobs(mask, n_blobs=1):
    """Return mask with only the N largest connected components kept."""
    if mask is None or not np.any(mask):
        return mask
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(-areas) + 1
    n_take = min(n_blobs, len(areas))
    keep = set(sorted_idx[:n_take].tolist())
    out = np.zeros_like(mask)
    for idx in keep:
        out[labels == idx] = True
    return out


# ═══════════════════════════════════════════════════════════════
#  VIDEO READER — 1 FPS SAMPLED
# ═══════════════════════════════════════════════════════════════
class VideoReader:
    def __init__(self, path, sample_fps=SAMPLE_FPS):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        self.orig_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.step = max(1, round(self.orig_fps / sample_fps))
        self.sample_map = list(range(0, self.orig_total, self.step))
        self.total = len(self.sample_map)
        self._lock = threading.Lock()
        self._cap  = cv2.VideoCapture(self.path)
        print(f"[video] {self.orig_total} orig @ {self.orig_fps:.1f} fps → {self.total} samples (step {self.step})")

    def get_frame(self, sample_idx):
        if sample_idx < 0 or sample_idx >= self.total: return None
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.sample_map[sample_idx])
            ok, f = self._cap.read()
        return f if ok else None

    def frame_jpeg(self, sample_idx, q=85):
        f = self.get_frame(sample_idx)
        if f is None: return None
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes()

    def extract_range_to_dir(self, start_sample, count, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        written = 0
        for i in range(count):
            si = start_sample + i
            if si >= self.total: break
            f = self.get_frame(si)
            if f is None: break
            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), f)
            written += 1
        return written

    def close(self): self._cap.release()

# ═══════════════════════════════════════════════════════════════
#  SAM2 TRACKER  (with all extended features)
# ═══════════════════════════════════════════════════════════════
def _load_predictor():
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        print(f"[sam2] Loading {SAM2_HF_ID} …")
        p = SAM2VideoPredictor.from_pretrained(SAM2_HF_ID)
        print("[sam2] Loaded ✓")
        return p
    except Exception as e:
        print(f"[sam2] from_pretrained failed: {e}")
    try:
        cfg  = globals().get("SAM2_CONFIG")
        ckpt = globals().get("SAM2_CHECKPOINT")
        if cfg and ckpt:
            from sam2.build_sam import build_sam2_video_predictor
            p = build_sam2_video_predictor(cfg, ckpt)
            print("[sam2] Loaded (local) ✓")
            return p
    except Exception as e:
        print(f"[sam2] local build failed: {e}")
    return None

def _autocast_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


class TrackerState:
    def __init__(self):
        self.masks    = {}   # sample_idx → {oid: bool ndarray}
        self.bboxes   = {}   # sample_idx → {oid: [x1,y1,x2,y2] or None}
        self.objects   = {}   # oid → {"label","box","n_blobs"}
        self.oof       = {}     # {oid: frame_from} — OOF from that frame onwards
        self.current_frame = 0
        self.start = self.end = None
        self.running = False
        self.phase = ""
        self.progress = self.total = 0
        self.error = None


class SAM2Tracker:
    def __init__(self):
        self.predictor = _load_predictor()
        self.state = TrackerState()
        self._tmp = None

    @property
    def available(self): return self.predictor is not None

    # ─── actions ──────────────────────────────────
    def start_tracking(self, video, sample_idx, boxes, duration_sec):
        if self.state.running: return
        s = self.state
        s.running = True; s.error = None
        next_id = max(s.objects.keys(), default=0) + 1
        for i, b in enumerate(boxes):
            s.objects[next_id + i] = {
                "label": b["label"], "box": b["box"],
                "n_blobs": int(b.get("n_blobs", 1)),
                "obj_start": None, "obj_end": None,
            }
        t = threading.Thread(target=self._run,
            args=(video, sample_idx, boxes, duration_sec, next_id), daemon=True)
        t.start()

    def reset(self):
        if self.state.running: return
        self.state = TrackerState()
        self._delete_autosave()

    def update_bbox(self, si, oid, bbox):
        s = self.state
        if si in s.bboxes and oid in s.bboxes[si]:
            s.bboxes[si][oid] = [int(v) for v in bbox]
            self._autosave_debounced()
            return True
        return False

    def relabel(self, oid, new_label):
        s = self.state
        if oid in s.objects:
            s.objects[oid]["label"] = new_label
            self._autosave()
            return True
        return False

    def toggle_oof(self, si, oid):
        """Toggle OOF: if OOF exists for this object, remove it. Otherwise set OOF from current frame onwards."""
        s = self.state
        if oid in s.oof:
            # Object has OOF marker - remove it to make visible
            del s.oof[oid]
            is_oof = False
        else:
            # No OOF marker - set OOF from current frame onwards
            s.oof[oid] = si
            is_oof = True
        self._autosave_debounced()
        return is_oof

    def update_nblobs(self, oid, n_blobs):
        s = self.state
        if oid not in s.objects: return False
        s.objects[oid]["n_blobs"] = n_blobs
        # Recompute masks (keep only n blobs) and bboxes
        for si, om in s.masks.items():
            if oid in om:
                filtered = mask_keep_n_blobs(om[oid], n_blobs)
                s.masks[si][oid] = filtered
                if si not in s.bboxes: s.bboxes[si] = {}
                s.bboxes[si][oid] = mask_to_bbox(filtered, n_blobs)
        self._autosave()
        return True

    def delete_object(self, oid):
        s = self.state
        if oid not in s.objects: return False
        del s.objects[oid]
        for si in list(s.masks.keys()):
            s.masks[si].pop(oid, None)
            s.bboxes.get(si, {}).pop(oid, None)
            # Clean up empty frames
            if not s.masks[si]: del s.masks[si]
            if si in s.bboxes and not s.bboxes[si]: del s.bboxes[si]
        s.oof.pop(oid, None)
        self._recalc_range()
        self._autosave()
        return True

    def _recalc_range(self):
        """Recalculate global start/end from per-object ranges."""
        s = self.state
        if not s.objects:
            s.start = s.end = None
            return
        starts, ends = [], []
        for oid, info in s.objects.items():
            os = info.get("obj_start")
            oe = info.get("obj_end")
            if os is not None: starts.append(os)
            if oe is not None: ends.append(oe)
        s.start = min(starts) if starts else None
        s.end = max(ends) if ends else None

    # ─── extend single object ─────────────────────
    def extend_object(self, video, oid, duration_sec):
        if self.state.running: return
        s = self.state
        s.running = True; s.error = None
        t = threading.Thread(target=self._run_extend,
            args=(video, oid, duration_sec), daemon=True)
        t.start()

    def _run_extend(self, video, oid, duration_sec):
        s = self.state
        try:
            obj_info = s.objects.get(oid)
            if not obj_info:
                s.error = f"Object {oid} not found"; return
            # Use per-object end, fall back to global
            start_from = obj_info.get("obj_end") or s.end
            if start_from is None:
                s.error = f"No end frame for object {oid}"; return

            # Find last bbox for this object (may not be exactly at start_from)
            bbox = s.bboxes.get(start_from, {}).get(oid)
            if bbox is None:
                for si in range(start_from, -1, -1):
                    bbox = s.bboxes.get(si, {}).get(oid)
                    if bbox is not None:
                        start_from = si; break
            if bbox is None:
                s.error = f"No bbox found for object {oid}"; return

            n_frames = min(duration_sec, video.total - start_from)
            s.total = n_frames
            new_end = start_from + n_frames - 1
            s.end = max(s.end, new_end) if s.end is not None else new_end

            s.phase = "extracting"
            self._tmp = tempfile.mkdtemp(prefix="sam2_")
            written = video.extract_range_to_dir(start_from, n_frames, self._tmp)
            print(f"[sam2] Extend obj {oid}: {written} frames from sample {start_from}")

            nb = obj_info.get("n_blobs", 2)

            s.phase = "embedding"
            dtype = _autocast_dtype()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                inf = self.predictor.init_state(video_path=self._tmp)
                box = np.array(bbox, dtype=np.float32)
                self.predictor.add_new_points_or_box(
                    inference_state=inf, frame_idx=0, obj_id=oid, box=box)

                s.phase = "tracking"
                for rel_idx, obj_ids, masks_t in self.predictor.propagate_in_video(inf):
                    abs_si = start_from + rel_idx
                    if abs_si not in s.masks: s.masks[abs_si] = {}
                    if abs_si not in s.bboxes: s.bboxes[abs_si] = {}
                    for i, o in enumerate(obj_ids):
                        o = int(o)
                        if o != oid: continue
                        m = (masks_t[i] > 0.0).cpu().numpy().squeeze().astype(bool)
                        s.masks[abs_si][o] = m
                        s.bboxes[abs_si][o] = mask_to_bbox(m, nb)
                    s.progress = rel_idx + 1

                self.predictor.reset_state(inf)

            # Update per-object end
            obj_info["obj_end"] = new_end

            n_total = sum(1 for si in s.bboxes if oid in s.bboxes[si] and s.bboxes[si][oid] is not None)
            print(f"[sam2] Extended obj {oid} → {n_total} total bboxes (end={new_end}) ✓")
            self._autosave()

        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            s.running = False
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None

    def _is_oof(self, si, oid):
        """Check if object is OOF at given frame."""
        return oid in self.state.oof and si >= self.state.oof[oid]

    # ─── queries ──────────────────────────────────
    def get_frame_data(self, si):
        """Return all object data for a frame (bboxes, oof status).
        Always includes all objects so frontend knows OOF status."""
        s = self.state
        if not s.objects: return None
        result = {}
        bb = s.bboxes.get(si, {})
        for oid, info in s.objects.items():
            is_oof = self._is_oof(si, oid)
            box_raw = bb.get(oid)
            box = box_raw
            if is_oof:
                box = None
            rgb = list(INST_COLOR.get(info["label"], (255,255,255)))
            result[str(oid)] = {
                "label": info["label"], "bbox": box, "bbox_raw": box_raw,
                "color": rgb, "oof": is_oof,
                "n_blobs": info.get("n_blobs", 1),
            }
        return result if result else None

    def bbox_ranges(self):
        """Return per-object bbox ranges, excluding OOF frames."""
        s = self.state
        if not s.bboxes: return {}
        frames_by_oid = {}
        for si in sorted(s.bboxes.keys()):
            ob = s.bboxes.get(si) or {}
            for oid, box in ob.items():
                if self._is_oof(si, oid): continue
                if box is None: continue
                frames_by_oid.setdefault(oid, []).append(si)
        ranges = {}
        for oid, frames in frames_by_oid.items():
            if not frames: continue
            frames.sort()
            start = prev = frames[0]
            spans = []
            for f in frames[1:]:
                if f == prev + 1:
                    prev = f; continue
                spans.append([start, prev])
                start = prev = f
            spans.append([start, prev])
            ranges[str(oid)] = spans
        return ranges

    def mask_png(self, si):
        """Return RGBA PNG overlay of masks for a frame."""
        s = self.state
        om = s.masks.get(si)
        if not om: return None
        first = next(iter(om.values()))
        h, w = first.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for oid, mask in om.items():
            if self._is_oof(si, oid): continue
            info = s.objects.get(oid, {})
            r, g, b = INST_COLOR.get(info.get("label",""), (255,255,255))
            rgba[mask, 0] = b  # BGRA for cv2
            rgba[mask, 1] = g
            rgba[mask, 2] = r
            rgba[mask, 3] = 180
        _, buf = cv2.imencode(".png", rgba)
        return buf.tobytes()

    # ─── export ───────────────────────────────────
    def export_video(self, video, path):
        s = self.state
        if not s.bboxes: return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
               "-s",f"{video.width}x{video.height}","-pix_fmt","bgr24",
               "-r","1","-i","-","-c:v","libx264","-pix_fmt","yuv420p",
               "-preset","fast",str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for si in range(s.start, s.end + 1):
            frame = video.get_frame(si)
            if frame is None: continue
            # draw mask overlay
            om = s.masks.get(si, {})
            overlay = frame.copy()
            for oid, mask in om.items():
                if self._is_oof(si, oid): continue
                info = s.objects.get(oid, {})
                c = np.array(INST_COLOR.get(info.get("label",""), (255,255,255))[::-1], dtype=np.float64)
                overlay[mask] = (overlay[mask]*0.6 + c*0.4).astype(np.uint8)
            frame = overlay
            # draw bboxes
            bb = s.bboxes.get(si, {})
            for oid, box in bb.items():
                if box is None or self._is_oof(si, oid): continue
                info = s.objects.get(oid, {})
                label = info.get("label", "?")
                r, g, b = INST_COLOR.get(label, (255,255,255))
                x1,y1,x2,y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1,y1),(x2,y2),(b,g,r),3)
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(y1-8, th+4)
                cv2.rectangle(frame,(x1,ly-th-4),(x1+tw+8,ly+4),(b,g,r),-1)
                cv2.putText(frame, label, (x1+4,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close(); proc.wait()
        return proc.returncode == 0

    def export_data(self, path):
        s = self.state
        if not s.masks and not s.bboxes: return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for si, om in sorted(s.masks.items()):
            for oid, mask in om.items():
                lbl = s.objects.get(oid, {}).get("label", "unk")
                arrays[f"s{si:06d}_{lbl}_o{oid}"] = mask
        if arrays:
            np.savez_compressed(path, **arrays)
        bbox_path = str(path).replace(".npz", "_bboxes.json")
        bbox_export = {}
        for si in sorted(s.bboxes.keys()):
            fb = {}
            for oid, box in s.bboxes[si].items():
                is_oof = self._is_oof(si, oid)
                lbl = s.objects.get(oid, {}).get("label", "unk")
                fb[str(oid)] = {
                    "label": lbl,
                    "bbox_xyxy": None if is_oof else box,
                    "out_of_frame": is_oof,
                }
            if fb: bbox_export[str(si)] = fb
        with open(bbox_path, "w") as f:
            json.dump({"objects": {str(k): v for k,v in s.objects.items()},
                        "start": s.start, "end": s.end,
                        "frames": bbox_export}, f, indent=2)
        return True

    # ─── autosave / autoload ──────────────────────
    _save_timer = None

    def _autosave_debounced(self):
        if self._save_timer: self._save_timer.cancel()
        self._save_timer = threading.Timer(1.0, self._autosave)
        self._save_timer.start()

    def _autosave(self):
        try:
            s = self.state
            AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
            bboxes_ser = {}
            for si, bb in s.bboxes.items():
                bboxes_ser[str(si)] = {str(oid): box for oid, box in bb.items()}
            oof_ser = {str(oid): frame_from for oid, frame_from in s.oof.items()}
            data = {
                "objects": {str(k): v for k, v in s.objects.items()},
                "bboxes": bboxes_ser,
                "oof": oof_ser,
                "start": s.start, "end": s.end,
                "current_frame": s.current_frame,
            }
            with open(AUTOSAVE_DIR / "state.json", "w") as f:
                json.dump(data, f)
            arrays = {}
            for si, om in s.masks.items():
                for oid, mask in om.items():
                    arrays[f"s{si:06d}_o{oid}"] = mask
            if arrays:
                np.savez_compressed(AUTOSAVE_DIR / "masks.npz", **arrays)
            print(f"[autosave] Saved {len(s.objects)} objects, {len(s.bboxes)} frames")
        except Exception as e:
            print(f"[autosave] Error: {e}")

    def _autoload(self):
        sp = AUTOSAVE_DIR / "state.json"
        if not sp.exists(): return False
        try:
            with open(sp) as f:
                data = json.load(f)
            s = self.state
            s.objects = {int(k): v for k, v in data["objects"].items()}
            s.start = data.get("start")
            s.end = data.get("end")
            s.current_frame = int(data.get("current_frame") or 0)
            for si_str, bb in data.get("bboxes", {}).items():
                si = int(si_str)
                s.bboxes[si] = {}
                for oid_str, box in bb.items():
                    s.bboxes[si][int(oid_str)] = box
            # Load OOF - handle both old format (list) and new format (dict)
            oof_data = data.get("oof", {})
            if isinstance(oof_data, list):
                # Old format: [[frame, oid], ...] - convert to new format (use min frame per oid)
                oof_by_oid = {}
                for si, oid in oof_data:
                    if oid not in oof_by_oid or si < oof_by_oid[oid]:
                        oof_by_oid[oid] = si
                s.oof = oof_by_oid
            else:
                # New format: {oid: frame_from}
                s.oof = {int(oid): frame_from for oid, frame_from in oof_data.items()}
            mask_path = AUTOSAVE_DIR / "masks.npz"
            if mask_path.exists():
                npz = np.load(mask_path)
                for key in npz.files:
                    parts = key.split("_o")
                    si = int(parts[0][1:])
                    oid = int(parts[1])
                    if si not in s.masks: s.masks[si] = {}
                    s.masks[si][oid] = npz[key]
            print(f"[autoload] Restored {len(s.objects)} objects, {len(s.bboxes)} frames")
            return True
        except Exception as e:
            print(f"[autoload] Error: {e}")
            return False

    def _delete_autosave(self):
        if AUTOSAVE_DIR.exists():
            shutil.rmtree(AUTOSAVE_DIR, ignore_errors=True)

    # ─── SAM2 tracking run ────────────────────────
    def _run(self, video, sample_idx, boxes, duration_sec, start_oid):
        s = self.state
        try:
            n_frames = min(duration_sec, video.total - sample_idx)
            s.total = n_frames
            new_start, new_end = sample_idx, sample_idx + n_frames - 1
            s.start = min(s.start, new_start) if s.start is not None else new_start
            s.end   = max(s.end, new_end) if s.end is not None else new_end

            # Phase 1 — extract frames to temp dir
            s.phase = "extracting"
            self._tmp = tempfile.mkdtemp(prefix="sam2_")
            written = video.extract_range_to_dir(sample_idx, n_frames, self._tmp)
            print(f"[sam2] Extracted {written} frames → {self._tmp}")

            # Build n_blobs map for bbox extraction
            n_blobs_map = {}
            for i, binfo in enumerate(boxes):
                n_blobs_map[start_oid + i] = int(binfo.get("n_blobs", 1))

            # Phase 2 — init state + add box prompts
            s.phase = "embedding"
            dtype = _autocast_dtype()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                inf = self.predictor.init_state(video_path=self._tmp)

                for i, binfo in enumerate(boxes):
                    oid = start_oid + i
                    box = np.array(binfo["box"], dtype=np.float32)
                    print(f"[sam2] obj {oid} ({binfo['label']}): box={box.tolist()}")
                    self.predictor.add_new_points_or_box(
                        inference_state=inf, frame_idx=0, obj_id=oid, box=box)

                # Phase 3 — propagate
                s.phase = "tracking"
                for rel_idx, obj_ids, masks_t in self.predictor.propagate_in_video(inf):
                    abs_si = sample_idx + rel_idx
                    if abs_si not in s.masks: s.masks[abs_si] = {}
                    if abs_si not in s.bboxes: s.bboxes[abs_si] = {}
                    for i, oid in enumerate(obj_ids):
                        oid = int(oid)
                        m = (masks_t[i] > 0.0).cpu().numpy().squeeze().astype(bool)
                        nb = n_blobs_map.get(oid, s.objects.get(oid, {}).get("n_blobs", 1))
                        s.masks[abs_si][oid] = m
                        s.bboxes[abs_si][oid] = mask_to_bbox(m, nb)
                    s.progress = rel_idx + 1

                self.predictor.reset_state(inf)

            n_total = sum(1 for bb in s.bboxes.values() for b in bb.values() if b is not None)
            # Set per-object tracking range
            for i in range(len(boxes)):
                oid = start_oid + i
                if oid in s.objects:
                    s.objects[oid]["obj_start"] = new_start
                    s.objects[oid]["obj_end"] = new_end
            print(f"[sam2] Done — {len(s.masks)} frames, {len(s.objects)} objects, {n_total} bboxes ✓")
            self._autosave()

        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            s.running = False
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None

# ═══════════════════════════════════════════════════════════════
#  HTML / JS
# ═══════════════════════════════════════════════════════════════
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surgical Instrument Tracker</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#0c1017;--panel:#151c28;--border:#1e2a3a;--border-h:#2a3a50;--text:#c8d6e5;--dim:#5a6d82;--accent:#00b4d8;--green:#06d6a0;--red:#ef476f;--orange:#ffd166}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.app{max-width:1500px;margin:auto;padding:12px 16px}
.header{display:flex;align-items:center;gap:12px;padding:8px 0 10px;border-bottom:1px solid var(--border);margin-bottom:10px}
.header h1{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:600;color:var(--accent);letter-spacing:.5px}
.pill{font-size:11px;padding:3px 10px;border-radius:99px;font-weight:500}
.pill-sam{background:#1a2636;color:var(--green);border:1px solid #1c3a2e}
.pill-sam.off{color:var(--red);border-color:#3a1c24}
.pill-fps{background:#1a2636;color:var(--orange);border:1px solid #3a3520}
.info-bar{display:flex;gap:20px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--dim);background:var(--panel);border:1px solid var(--border);padding:7px 14px;border-radius:6px;margin-bottom:8px}
.info-bar b{color:var(--text);font-weight:500}
.label-selector{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.label-pill{font-size:12px;font-weight:600;padding:6px 14px;border-radius:99px;border:2px solid transparent;cursor:pointer;transition:all .15s;opacity:.55;background:var(--panel);color:var(--text)}
.label-pill:hover{opacity:.8}
.label-pill.active{opacity:1;border-color:var(--clr);box-shadow:0 0 8px color-mix(in srgb,var(--clr) 40%,transparent)}
.nav{display:flex;gap:5px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.nav button,.nav input{font-family:'JetBrains Mono',monospace;font-size:12px}
.nav button{background:var(--panel);color:var(--text);border:1px solid var(--border);padding:5px 10px;border-radius:4px;cursor:pointer;transition:all .15s}
.nav button:hover{border-color:var(--border-h);background:#1a2535}
.nav input[type=number]{width:80px;padding:5px 8px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;text-align:center}
.nav .shortcuts{font-size:11px;color:var(--dim);margin-left:8px}
.main-area{display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap}
.canvas-wrap{position:relative;display:inline-block;border:1px solid var(--border);border-radius:6px;overflow:hidden;background:#000;line-height:0;flex-shrink:0}
#canvas{display:block;max-width:50vw;cursor:crosshair}
.side-panel{min-width:240px;max-width:320px;flex:1}
.shdr{font-size:12px;font-weight:600;color:var(--dim);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.blist{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:8px;max-height:200px;overflow-y:auto;margin-bottom:10px}
.bitem{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;font-size:12px;font-family:'JetBrains Mono',monospace}
.bitem:hover{background:#1a2535}
.bdot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.bitem .lbl{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bitem input[type=number]{width:40px;padding:2px 4px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;font-size:11px;text-align:center}
.bitem button,.tobj button{background:none;border:none;color:var(--red);cursor:pointer;font-size:14px;padding:0 3px;opacity:.6}
.bitem button:hover,.tobj button:hover{opacity:1}
.bempty{color:var(--dim);font-size:12px;font-style:italic}
.tobj{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;font-size:12px}
.tobj .oid{font-size:10px;color:var(--dim);font-family:'JetBrains Mono',monospace}
.tobj:hover{background:#1a2535}
.tobj select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:11px;font-family:'JetBrains Mono',monospace;cursor:pointer}
.tobj .oof-btn{font-size:10px;padding:2px 6px;border:1px solid var(--border);border-radius:3px;cursor:pointer;background:var(--bg);color:var(--dim);transition:all .15s}
.tobj .oof-btn.active{background:#3a1c24;color:var(--red);border-color:var(--red)}
.tobj .oof-btn:hover{border-color:var(--border-h)}
.tobj .ext-btn{font-size:10px;padding:2px 6px;border:1px solid #0a5c47;border-radius:3px;cursor:pointer;background:#073b2e;color:var(--green);transition:all .15s;font-weight:600}
.tobj .ext-btn:hover{background:#0a5240}
.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.actions button{font-family:'IBM Plex Sans',sans-serif;font-size:13px;font-weight:500;padding:7px 16px;border-radius:5px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;transition:all .15s}
.actions button:hover{border-color:var(--border-h);background:#1a2535}
.actions button:disabled{opacity:.35;cursor:not-allowed}
.btn-track{background:#073b2e!important;border-color:#0a5c47!important;color:var(--green)!important}
.btn-track:hover:not(:disabled){background:#0a5240!important}
.btn-danger{color:var(--red)!important;border-color:#3a1c24!important}
.btn-export{color:var(--orange)!important;border-color:#3a3520!important}
.actions label{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:6px}
.actions input[type=number]{width:65px;padding:4px 6px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:12px}
.actions input[type=checkbox]{accent-color:var(--accent)}
.pbar-wrap{flex:1;min-width:200px;display:flex;align-items:center;gap:10px}
.pbar{flex:1;height:10px;background:var(--panel);border:1px solid var(--border);border-radius:99px;overflow:hidden}
.pfill{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,#0077b6,var(--accent));transition:width .3s}
.ptxt{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--dim);white-space:nowrap}
.timeline{height:28px;background:var(--panel);border:1px solid var(--border);border-radius:5px;position:relative;cursor:pointer;overflow:hidden}
.tl-lanes{position:absolute;inset:0}
.tl-lane{position:absolute;left:0;right:0}
.tl-seg{position:absolute;top:2px;bottom:2px;border-radius:4px;opacity:.45}
.tl-seg:hover{opacity:.70}
.tl-cur{position:absolute;top:0;width:2px;height:100%;background:var(--red);pointer-events:none}
.hidden{display:none!important}
.toast{position:fixed;bottom:20px;right:20px;background:#1a2636;border:1px solid var(--border);padding:10px 18px;border-radius:8px;font-size:13px;color:var(--green);transform:translateY(80px);opacity:0;transition:all .3s;z-index:99}
.toast.show{transform:translateY(0);opacity:1}
.toast.err{color:var(--red)}
.lbl-dropdown{position:absolute;z-index:100;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:12px;font-family:'JetBrains Mono',monospace;cursor:pointer}
.lbl-dropdown.hidden{display:none}
</style></head>
<body>
<div class="app">
  <div class="header">
    <h1>▸ INSTRUMENT TRACKER</h1>
    <span class="pill pill-sam" id="samPill">SAM2: …</span>
    <span class="pill pill-fps" id="fpsPill">1 FPS</span>
  </div>
  <div class="info-bar">
    <span>sample <b id="iFrame">0</b> / <b id="iTotal">?</b></span>
    <span>time <b id="iTime">0s</b></span>
    <span>res <b id="iRes">?</b></span>
    <span>1 sample = 1 sec</span>
  </div>
  <div class="label-selector" id="labelSel"></div>
  <div class="nav">
    <button onclick="jmp(-60)">-60s</button><button onclick="jmp(-10)">-10s</button>
    <button onclick="jmp(-1)">◂ -1s</button>
    <input type="number" id="fIn" value="0" min="0">
    <button onclick="goF()">go</button>
    <button onclick="jmp(1)">+1s ▸</button><button onclick="jmp(10)">+10s</button>
    <button onclick="jmp(60)">+60s</button>
    <span style="width:12px"></span>
    <button id="playBtn" onclick="togPlay()">▶ play</button>
    <span class="shortcuts">A/D ±1 · W/S ±10 · Q/E ±60 · Space=play</span>
  </div>
  <div class="main-area">
    <div class="canvas-wrap"><canvas id="canvas" width="800" height="450"></canvas></div>
    <div class="side-panel">
      <div class="shdr">New Boxes (to track)</div>
      <div class="blist" id="boxList"><span class="bempty">Select instrument → draw on frame</span></div>
      <div id="trkWrap" class="hidden">
        <div class="shdr">Tracked Objects</div>
        <div class="blist" id="trkList"></div>
      </div>
      <div id="oofWrap" class="hidden">
        <div class="shdr">Out of Frame</div>
        <div class="blist" id="oofList"></div>
      </div>
    </div>
  </div>
  <div class="actions">
    <button class="btn-danger" onclick="clearDrawn()">clear drawn</button>
    <button class="btn-track" id="trackBtn" onclick="startTrack()" disabled>▸ track</button>
    <label>dur <input type="number" id="durIn" value="50" min="5" max="3600">s</label>
    <label id="cbMaskLbl" class="hidden"><input type="checkbox" id="cbMask" checked onchange="redr()"> masks</label>
    <label id="cbBoxLbl" class="hidden"><input type="checkbox" id="cbBox" checked onchange="redr()"> boxes</label>
    <button id="expVid" class="btn-export hidden" onclick="doExp('video')">export video</button>
    <button id="expDat" class="btn-export hidden" onclick="doExp('data')">export data</button>
    <button id="rstBtn" class="btn-danger hidden" onclick="rstAll()">reset all</button>
  </div>
  <div class="actions hidden" id="progRow">
    <div class="pbar-wrap"><div class="pbar"><div class="pfill" id="pFill"></div></div>
    <span class="ptxt" id="pTxt">starting…</span></div>
  </div>
  <div class="timeline" id="tl" onclick="tlClk(event)">
    <div class="tl-lanes" id="tlLanes"></div>
    <div class="tl-cur" id="tlCur" style="left:0"></div>
  </div>
</div>
<div class="toast" id="toast"></div>
<select id="lblDropdown" class="lbl-dropdown hidden"></select>

<script>
var S={frame:0,total:0,vw:0,vh:0,boxes:[],activeLabel:'',activeColor:'',
  instruments:[],tracked:null,fdCache:{},maskCache:{},tracking:false,playTimer:null,
  allObjects:{},labelRects:[],bboxRanges:null,bboxRangesInFlight:false};
var canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
var frameImg=null,maskImg=null,drag=null;

fetch('/api/info').then(r=>r.json()).then(d=>{
  S.total=d.total_samples;S.vw=d.width;S.vh=d.height;S.instruments=d.instruments;
  document.getElementById('iTotal').textContent=S.total;
  document.getElementById('iRes').textContent=d.width+'×'+d.height;
  document.getElementById('fpsPill').textContent=d.sample_fps+' FPS';
  var sp=document.getElementById('samPill');
  sp.textContent=d.sam2_ok?'SAM2: ready':'SAM2: not found';
  sp.classList.toggle('off',!d.sam2_ok);
  var sel=document.getElementById('labelSel');
  d.instruments.forEach((inst,i)=>{
    var b=document.createElement('button');
    b.className='label-pill'+(i===0?' active':'');
    b.dataset.label=inst.name;b.style.setProperty('--clr',inst.hex);b.style.color=inst.hex;
    b.textContent=inst.name;b.onclick=()=>selLbl(inst.name,inst.hex);sel.appendChild(b);
  });
  if(d.instruments.length){S.activeLabel=d.instruments[0].name;S.activeColor=d.instruments[0].hex;}
  if(d.has_tracked){
    S.tracked={start:d.tracked_start,end:d.tracked_end};
    S.allObjects=d.objects||{};
    S.bboxRanges=null;
    updTimeline();updUI();
  }
  var lf=(d.current_frame!=null)?d.current_frame:((d.has_tracked && d.tracked_start!=null)?d.tracked_start:0);
  loadFrame(lf);
});

function selLbl(n,h){S.activeLabel=n;S.activeColor=h;
  document.querySelectorAll('.label-pill').forEach(b=>b.classList.toggle('active',b.dataset.label===n));}
function instHex(l){var i=S.instruments.find(x=>x.name===l);return i?i.hex:'#fff';}
function rgb(a){return 'rgb('+a[0]+','+a[1]+','+a[2]+')';}

function loadImg(u){return new Promise((r,j)=>{var i=new Image();i.onload=()=>r(i);i.onerror=j;i.src=u;});}
var _loadId=0;
var _saveFrameTimer=null;
function saveFrameDebounced(){
  if(_saveFrameTimer)clearTimeout(_saveFrameTimer);
  _saveFrameTimer=setTimeout(()=>{fetch('/api/set_frame',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame})});},400);
}
async function loadFrame(n){
  n=Math.max(0,Math.min(n,S.total-1));S.frame=n;
  saveFrameDebounced();
  document.getElementById('fIn').value=n;
  document.getElementById('iFrame').textContent=n;
  document.getElementById('iTime').textContent=n+'s';
  document.getElementById('tlCur').style.left=(S.total>1?(n/(S.total-1))*100:0)+'%';
  var myId=++_loadId;
  var newFrame;
  try{newFrame=await loadImg('/api/frame/'+n+'?t='+Date.now());}catch(e){return;}
  if(_loadId!==myId)return;/* superseded by newer loadFrame call */
  var newMask=null;
  if(S.tracked){
    if(S.fdCache[n]===undefined){
      try{var r=await fetch('/api/framedata/'+n);if(_loadId!==myId)return;S.fdCache[n]=r.ok?await r.json():null;}
      catch(e){S.fdCache[n]=null;}}
    if(n>=S.tracked.start&&n<=S.tracked.end){
      if(S.maskCache[n]===undefined){
        try{newMask=await loadImg('/api/mask/'+n);if(_loadId!==myId)return;S.maskCache[n]=newMask;}
        catch(e){S.maskCache[n]=null;}}
      else newMask=S.maskCache[n];
    }
  }
  if(_loadId!==myId)return;
  frameImg=newFrame;maskImg=newMask;
  if(canvas.width!==frameImg.naturalWidth)canvas.width=frameImg.naturalWidth;
  if(canvas.height!==frameImg.naturalHeight)canvas.height=frameImg.naturalHeight;
  redr();updTrkList();updTimeline();
}

var HZ=14;
function hitTest(x,y){
  if(!document.getElementById('cbBox').checked)return null;
  var fd=S.fdCache[S.frame];if(!fd)return null;
  var oids=Object.keys(fd);
  for(var k=oids.length-1;k>=0;k--){
    var oid=oids[k],o=fd[oid];if(!o||!o.bbox||o.oof)continue;
    var b=o.bbox,x1=b[0],y1=b[1],x2=b[2],y2=b[3],H=HZ;
    if(Math.abs(x-x1)<H&&Math.abs(y-y1)<H)return{type:'resize',oid,handle:'tl',cursor:'nwse-resize'};
    if(Math.abs(x-x2)<H&&Math.abs(y-y1)<H)return{type:'resize',oid,handle:'tr',cursor:'nesw-resize'};
    if(Math.abs(x-x1)<H&&Math.abs(y-y2)<H)return{type:'resize',oid,handle:'bl',cursor:'nesw-resize'};
    if(Math.abs(x-x2)<H&&Math.abs(y-y2)<H)return{type:'resize',oid,handle:'br',cursor:'nwse-resize'};
    if(Math.abs(y-y1)<H&&x>x1+H&&x<x2-H)return{type:'resize',oid,handle:'t',cursor:'ns-resize'};
    if(Math.abs(y-y2)<H&&x>x1+H&&x<x2-H)return{type:'resize',oid,handle:'b',cursor:'ns-resize'};
    if(Math.abs(x-x1)<H&&y>y1+H&&y<y2-H)return{type:'resize',oid,handle:'l',cursor:'ew-resize'};
    if(Math.abs(x-x2)<H&&y>y1+H&&y<y2-H)return{type:'resize',oid,handle:'r',cursor:'ew-resize'};
  }
  return null;
}
function dragBox(px,py){
  if(!drag||drag.type==='draw')return null;
  var ob=drag.orig,dx=px-drag.sx,dy=py-drag.sy,nb;
  if(drag.type==='move'){nb=[ob[0]+dx,ob[1]+dy,ob[2]+dx,ob[3]+dy];}
  else{nb=ob.slice();var h=drag.handle;
    if('tl l bl'.includes(h))nb[0]=ob[0]+dx;if('tl t tr'.includes(h))nb[1]=ob[1]+dy;
    if('tr r br'.includes(h))nb[2]=ob[2]+dx;if('bl b br'.includes(h))nb[3]=ob[3]+dy;
    if(nb[2]-nb[0]<10){nb[0]=ob[0];nb[2]=ob[2];}if(nb[3]-nb[1]<10){nb[1]=ob[1];nb[3]=ob[3];}}
  return nb;
}
function cCoords(e){var r=canvas.getBoundingClientRect();
  return{x:(e.clientX-r.left)*(canvas.width/r.width),y:(e.clientY-r.top)*(canvas.height/r.height)};}
function cScreenCoords(cx,cy){var r=canvas.getBoundingClientRect();
  return{x:r.left+cx*(r.width/canvas.width),y:r.top+cy*(r.height/canvas.height)};}
function hitTestLabel(x,y){
  for(var i=S.labelRects.length-1;i>=0;i--){
    var lr=S.labelRects[i];
    if(x>=lr.x&&x<=lr.x+lr.w&&y>=lr.y&&y<=lr.y+lr.h)return lr;
  }
  return null;
}
function showLabelDropdown(lr){
  var dd=document.getElementById('lblDropdown');
  dd.innerHTML='';
  S.instruments.forEach(inst=>{
    var opt=document.createElement('option');
    opt.value=inst.name;opt.textContent=inst.name;
    if(inst.name===lr.label)opt.selected=true;
    dd.appendChild(opt);
  });
  dd.dataset.oid=lr.oid;
  var sc=cScreenCoords(lr.x,lr.y+lr.h);
  dd.style.left=sc.x+'px';dd.style.top=sc.y+'px';
  dd.classList.remove('hidden');
  dd.focus();
}
document.getElementById('lblDropdown').addEventListener('change',async function(){
  var oid=parseInt(this.dataset.oid),newLabel=this.value;
  this.classList.add('hidden');
  await fetch('/api/relabel',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,label:newLabel})});
  S.allObjects[oid].label=newLabel;S.fdCache={};S.maskCache={};loadFrame(S.frame);
  toast('Relabeled obj '+oid+' → '+newLabel);
});
document.getElementById('lblDropdown').addEventListener('blur',function(){this.classList.add('hidden');});

canvas.addEventListener('mousedown',e=>{
  document.getElementById('lblDropdown').classList.add('hidden');
  if(S.tracking)return;var p=cCoords(e);
  var lblHit=hitTestLabel(p.x,p.y);
  if(lblHit){showLabelDropdown(lblHit);return;}
  var hit=hitTest(p.x,p.y);
  if(hit){var bb=S.fdCache[S.frame][hit.oid].bbox;
    drag={type:hit.type,oid:hit.oid,handle:hit.handle,sx:p.x,sy:p.y,orig:bb.slice(),box:bb.slice()};}
  else drag={type:'draw',sx:p.x,sy:p.y,box:null};
});
canvas.addEventListener('mousemove',e=>{
  var p=cCoords(e);
  if(!drag){
    var lblHit=hitTestLabel(p.x,p.y);
    if(lblHit){canvas.style.cursor='pointer';return;}
    var h=hitTest(p.x,p.y);canvas.style.cursor=h?h.cursor:'crosshair';return;}
  if(drag.type==='draw')drag.box=[Math.min(drag.sx,p.x),Math.min(drag.sy,p.y),Math.max(drag.sx,p.x),Math.max(drag.sy,p.y)];
  else drag.box=dragBox(p.x,p.y);
  redr();
});
canvas.addEventListener('mouseup',e=>{
  if(!drag)return;
  if(drag.type==='draw'){var b=drag.box;
    if(b&&(b[2]-b[0]>10)&&(b[3]-b[1]>10)){
      S.boxes.push({x1:b[0],y1:b[1],x2:b[2],y2:b[3],label:S.activeLabel,color:S.activeColor,n_blobs:2});
      updBoxList();}}
  else if(drag.box){S.fdCache[S.frame][drag.oid].bbox=drag.box;
    fetch('/api/update_bbox',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sample:S.frame,oid:parseInt(drag.oid),bbox:drag.box.map(v=>Math.round(v))})});}
  drag=null;updUI();redr();
});
canvas.addEventListener('mouseleave',()=>{if(drag&&drag.type!=='draw'){drag=null;redr();}});

function redr(){
  if(!frameImg)return;ctx.drawImage(frameImg,0,0);
  S.labelRects=[];/* clear label hit areas */
  if(maskImg&&document.getElementById('cbMask').checked){
    ctx.globalAlpha=0.5;ctx.drawImage(maskImg,0,0);ctx.globalAlpha=1;}
  if(document.getElementById('cbBox').checked){
    var fd=S.fdCache[S.frame];
    if(fd){for(var oid in fd){var o=fd[oid];if(!o||!o.bbox||o.oof)continue;
      var box=o.bbox;if(drag&&drag.oid===oid&&drag.box)box=drag.box;
      drawTBox(box[0],box[1],box[2],box[3],rgb(o.color),o.label,o.color,oid);}}}
  S.boxes.forEach(b=>drawNBox(b.x1,b.y1,b.x2,b.y2,b.color,b.label));
  if(drag&&drag.type==='draw'&&drag.box){var db=drag.box;drawNBox(db[0],db[1],db[2],db[3],S.activeColor,S.activeLabel);}
}
function drawTBox(x1,y1,x2,y2,col,lbl,ca,oid){
  ctx.strokeStyle=col;ctx.lineWidth=3;ctx.setLineDash([]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.fillStyle='rgba('+ca[0]+','+ca[1]+','+ca[2]+',0.08)';ctx.fillRect(x1,y1,x2-x1,y2-y1);
  ctx.font='bold 13px IBM Plex Sans,sans-serif';var tw=ctx.measureText(lbl).width+10,th=20;
  var ly=y1>th+4?y1-th-2:y2+2;ctx.fillStyle=col;ctx.fillRect(x1,ly,tw,th);
  ctx.fillStyle='#fff';ctx.fillText(lbl,x1+5,ly+15);
  /* store label rect for click detection */
  S.labelRects.push({x:x1,y:ly,w:tw,h:th,oid:oid,label:lbl});
  var hs=7;ctx.fillStyle=col;
  [[x1,y1],[x2,y1],[x1,y2],[x2,y2],[(x1+x2)/2,y1],[(x1+x2)/2,y2],[x1,(y1+y2)/2],[x2,(y1+y2)/2]]
    .forEach(c=>ctx.fillRect(c[0]-hs/2,c[1]-hs/2,hs,hs));
}
function drawNBox(x1,y1,x2,y2,col,lbl){
  ctx.strokeStyle=col;ctx.lineWidth=2;ctx.setLineDash([8,5]);
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
  ctx.font='bold 13px IBM Plex Sans,sans-serif';ctx.fillStyle=col;
  ctx.fillText('⊕ '+lbl,x1+4,y1>22?y1-6:y2+16);
}

function updBoxList(){
  var el=document.getElementById('boxList');
  if(!S.boxes.length){el.innerHTML='<span class="bempty">Select instrument → draw on frame</span>';return;}
  el.innerHTML=S.boxes.map((b,i)=>
    '<div class="bitem"><span class="bdot" style="background:'+b.color+'"></span>'+
    '<span class="lbl">'+b.label+'</span>'+
    '<label style="font-size:11px;color:var(--dim)">blobs:</label>'+
    '<input type="number" value="'+b.n_blobs+'" min="1" max="20" onchange="S.boxes['+i+'].n_blobs=+this.value">'+
    '<button onclick="rmBox('+i+')">×</button></div>').join('');
}
function rmBox(i){S.boxes.splice(i,1);updBoxList();updUI();redr();}
function clearDrawn(){S.boxes=[];updBoxList();updUI();redr();}

function updTrkList(){
  var trkEl=document.getElementById('trkList');
  var oofEl=document.getElementById('oofList');
  var fd=S.fdCache[S.frame];
  var objs=S.allObjects;
  if(!objs||!Object.keys(objs).length){
    trkEl.innerHTML='<span class="bempty">No tracked objects</span>';
    oofEl.innerHTML='';document.getElementById('oofWrap').classList.add('hidden');return;}
  var hTrk='',hOof='';
  for(var oid in objs){
    var o=objs[oid],c=instHex(o.label);
    var hasMask=!!(fd&&fd[oid]&&fd[oid].bbox_raw);
    var isOof=!!(fd&&fd[oid]&&fd[oid].oof);
    var nb=o.n_blobs||2;
    var objEnd=o.obj_end!=null?o.obj_end:(S.tracked?S.tracked.end:null);
    var atObjEnd=objEnd!=null&&S.frame>=objEnd;
    var row='<div class="tobj"><span class="bdot" style="background:'+c+'"></span>';
    row+='<span class="oid">#'+oid+'</span>';
    row+='<select onchange="relabel('+oid+',this.value)">';
    S.instruments.forEach(inst=>{
      row+='<option value="'+inst.name+'"'+(inst.name===o.label?' selected':'')+'>'+inst.name+'</option>';});
    row+='</select>';
    row+='<label style="font-size:10px;color:var(--dim)">n:</label>';
    row+='<input type="number" value="'+nb+'" min="1" max="20" style="width:36px;padding:1px 3px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;font-size:10px;text-align:center" onchange="updNblobs('+oid+',+this.value)">';
    if(isOof){
      row+='<span class="oof-btn active" onclick="togOof('+oid+')">OOF</span>';
    }else{
      row+='<span class="oof-btn" onclick="togOof('+oid+')">vis</span>';
      if(atObjEnd)row+='<span class="ext-btn" onclick="extendObj('+oid+')" title="Extend tracking from frame '+objEnd+'">▸trk</span>';
    }
    row+='<button onclick="delObj('+oid+')" title="Delete entire series">×</button></div>';
    if(isOof)hOof+=row;
    else if(hasMask)hTrk+=row;
  }
  trkEl.innerHTML=hTrk||'<span class="bempty">All objects out of frame</span>';
  oofEl.innerHTML=hOof;
  document.getElementById('oofWrap').classList.toggle('hidden',!hOof);
}

function updUI(){
  var hb=S.boxes.length>0,ht=S.tracked!==null;
  document.getElementById('trackBtn').disabled=!hb||S.tracking;
  ['cbMaskLbl','cbBoxLbl','expVid','expDat','rstBtn'].forEach(id=>
    document.getElementById(id).classList.toggle('hidden',!ht));
  document.getElementById('trkWrap').classList.toggle('hidden',!ht);
  document.getElementById('progRow').classList.toggle('hidden',!S.tracking);
}

function jmp(d){loadFrame(S.frame+d);}
function goF(){loadFrame(+document.getElementById('fIn').value||0);}
function togPlay(){
  if(S.playTimer){clearInterval(S.playTimer);S.playTimer=null;
    document.getElementById('playBtn').textContent='▶ play';return;}
  S.playTimer=setInterval(()=>{if(S.frame>=S.total-1){togPlay();return;}loadFrame(S.frame+1);},1000);
  document.getElementById('playBtn').textContent='⏸ stop';
}
function tlClk(e){var r=e.currentTarget.getBoundingClientRect();
  loadFrame(Math.round(((e.clientX-r.left)/r.width)*(S.total-1)));}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  var m={d:1,ArrowRight:1,a:-1,ArrowLeft:-1,w:10,ArrowUp:10,s:-10,ArrowDown:-10,q:-60,e:60};
  if(m[e.key]!==undefined){e.preventDefault();jmp(m[e.key]);}
  if(e.key===' '){e.preventDefault();togPlay();}
});

async function startTrack(){
  if(!S.boxes.length||S.tracking)return;
  var dur=+document.getElementById('durIn').value||50;
  S.tracking=true;updUI();
  var bd=S.boxes.map(b=>({box:[b.x1,b.y1,b.x2,b.y2],label:b.label,n_blobs:b.n_blobs||1}));
  var r=await fetch('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame,boxes:bd,duration:dur})});
  if(!r.ok){toast('Failed',true);S.tracking=false;updUI();return;}
  pollProg();
}
async function pollProg(){
  try{var r=await fetch('/api/progress');var d=await r.json();
    var fill=d.total>0?(d.progress/d.total)*100:0;
    document.getElementById('pFill').style.width=fill+'%';
    var t=d.phase||'starting';
    if(d.phase==='tracking')t='tracking '+d.progress+'/'+d.total+' sec';
    else if(d.phase==='embedding')t='computing embeddings…';
    else if(d.phase==='extracting')t='extracting frames…';
    document.getElementById('pTxt').textContent=t;
    if(d.error){toast(d.error,true);S.tracking=false;updUI();return;}
    if(d.running){setTimeout(pollProg,400);return;}
    S.tracking=false;S.tracked={start:d.start,end:d.end};
    S.fdCache={};S.maskCache={};S.boxes=[];S.allObjects=d.objects||{};S.bboxRanges=null;
    updBoxList();updTimeline();updUI();
    toast('Tracked '+(d.end-d.start+1)+'s — '+Object.keys(d.objects).length+' objects');
    loadFrame(S.frame);
  }catch(e){setTimeout(pollProg,1000);}
}
function fetchBboxRanges(){
  if(S.bboxRangesInFlight)return;
  S.bboxRangesInFlight=true;
  fetch('/api/bbox_ranges').then(r=>r.json()).then(d=>{
    S.bboxRanges=d.ranges||{};
    S.bboxRangesInFlight=false;
    updTimeline();
  }).catch(()=>{S.bboxRangesInFlight=false;});
}

function updTimeline(){
  var lanesEl=document.getElementById('tlLanes')
  var tl=document.getElementById('tl')

  if(!lanesEl || !tl)return

  lanesEl.innerHTML=''
  if(!S.tracked || !S.allObjects || !Object.keys(S.allObjects).length){
    tl.style.height='28px'
    return
  }

  if(!S.bboxRanges){
    tl.style.height='28px'
    fetchBboxRanges();
    return
  }

  var intervals=[]
  for(var oid in S.bboxRanges){
    var ranges=S.bboxRanges[oid]||[]
    if(!ranges.length)continue
    var o=S.allObjects[oid] || {}
    for(var ri=0;ri<ranges.length;ri++){
      var st=ranges[ri][0],en=ranges[ri][1]
      if(st==null || en==null)continue

      st=Math.max(0,Math.min(st,S.total-1))
      en=Math.max(0,Math.min(en,S.total-1))
      if(en<st)continue

      intervals.push({
        oid:String(oid),
        label:o.label || '',
        start:st,
        end:en,
        color:instHex(o.label || '')
      })
    }
  }

  if(!intervals.length){
    tl.style.height='28px'
    return
  }

  intervals.sort((a,b)=> (a.start-b.start) || (a.end-b.end))

  var laneEnds=[]
  for(var i=0;i<intervals.length;i++){
    var seg=intervals[i]
    var lane=-1
    for(var k=0;k<laneEnds.length;k++){
      if(seg.start>laneEnds[k]){lane=k;break}
    }
    if(lane<0){
      lane=laneEnds.length
      laneEnds.push(seg.end)
    }else{
      laneEnds[lane]=seg.end
    }
    seg.lane=lane
  }

  var nLanes=laneEnds.length
  var lanePx=14
  var padPx=6
  var h=Math.max(28, nLanes*lanePx + padPx)
  tl.style.height=h+'px'

  for(var li=0;li<nLanes;li++){
    var laneDiv=document.createElement('div')
    laneDiv.className='tl-lane'
    laneDiv.style.top=(li*lanePx)+'px'
    laneDiv.style.height=lanePx+'px'
    lanesEl.appendChild(laneDiv)
  }

  for(var j=0;j<intervals.length;j++){
    var s=intervals[j]
    var leftPct=(s.start / Math.max(S.total,1)) * 100
    var widthPct=((s.end + 1 - s.start) / Math.max(S.total,1)) * 100

    var segDiv=document.createElement('div')
    segDiv.className='tl-seg'
    segDiv.style.left=leftPct+'%'
    segDiv.style.width=widthPct+'%'
    segDiv.style.background=s.color
    segDiv.style.border='1px solid '+s.color
    segDiv.title=(s.label? s.label+' ' : '') + 'id ' + s.oid + '  ' + s.start + ' to ' + s.end

    var laneHost=lanesEl.children[s.lane]
    if(laneHost)laneHost.appendChild(segDiv)
  }
}

async function relabel(oid,newLabel){
  await fetch('/api/relabel',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,label:newLabel})});
  S.allObjects[oid].label=newLabel;S.fdCache={};S.maskCache={};updTimeline();loadFrame(S.frame);
  toast('Relabeled obj '+oid+' → '+newLabel);
}
async function togOof(oid){
  var r=await fetch('/api/toggle_oof',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sample:S.frame,oid:oid})});
  var d=await r.json();
  S.fdCache={};S.maskCache={};S.bboxRanges=null;
  await loadFrame(S.frame);
  updTimeline();
}
async function updNblobs(oid,n){
  await fetch('/api/update_nblobs',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,n_blobs:n})});
  S.allObjects[oid].n_blobs=n;S.fdCache={};S.maskCache={};S.bboxRanges=null;updTimeline();loadFrame(S.frame);
  toast('Obj '+oid+' → '+n+' blobs');
}
async function extendObj(oid){
  if(S.tracking)return;
  var dur=+document.getElementById('durIn').value||50;
  S.tracking=true;updUI();
  var r=await fetch('/api/extend_track',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,duration:dur})});
  if(!r.ok){toast('Failed',true);S.tracking=false;updUI();return;}
  pollProg();
}
async function delObj(oid){
  if(!confirm('Delete object '+oid+' from ALL frames?'))return;
  await fetch('/api/delete_object',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid})});
  delete S.allObjects[String(oid)];S.fdCache={};S.maskCache={};
  /* refresh range from server */
  var pr=await fetch('/api/progress');var pd=await pr.json();
  S.allObjects=pd.objects||{};
  if(!Object.keys(S.allObjects).length){S.tracked=null;}
  else{S.tracked={start:pd.start,end:pd.end};}
  S.bboxRanges=null;
  updTimeline();loadFrame(S.frame);updUI();
  toast('Deleted obj '+oid);
}
function rstAll(){
  if(!confirm('Reset ALL tracking data?'))return;
  fetch('/api/reset',{method:'POST'});
  S.tracked=null;S.fdCache={};S.maskCache={};S.boxes=[];S.allObjects={};S.bboxRanges=null;
  updBoxList();updTimeline();updUI();document.getElementById('trkList').innerHTML='';
  document.getElementById('oofList').innerHTML='';document.getElementById('oofWrap').classList.add('hidden');redr();
  toast('Reset');
}

async function doExp(t){toast('Exporting…');
  try{var r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({type:t})});var d=await r.json();
    d.ok?toast('Exported → '+d.path):toast(d.error||'fail',true);}catch(e){toast('fail',true);}}

var tt;function toast(m,e){var el=document.getElementById('toast');el.textContent=m;
  el.classList.toggle('err',!!e);el.classList.add('show');clearTimeout(tt);
  tt=setTimeout(()=>el.classList.remove('show'),4000);}
</script>
</body></html>
"""

# ═══════════════════════════════════════════════════════════════
#  HTTP SERVER
# ═══════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    video:   VideoReader  = None
    tracker: SAM2Tracker  = None

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._send(200, "text/html", HTML_PAGE.encode())
        elif p == "/api/info":
            s = self.tracker.state
            has_tracked = s.start is not None
            self._json({
                "total_samples": self.video.total,
                "sample_fps": SAMPLE_FPS,
                "step": self.video.step,
                "orig_fps": self.video.orig_fps,
                "width": self.video.width,
                "height": self.video.height,
                "sam2_ok": self.tracker.available,
                "instruments": INSTRUMENTS,
                "has_tracked": has_tracked,
                "tracked_start": s.start,
                "tracked_end": s.end,
                "current_frame": s.current_frame,
                "objects": {str(k): v for k, v in s.objects.items()} if has_tracked else {},
            })
        elif p.startswith("/api/frame/"):
            d = self.video.frame_jpeg(int(p.split("/")[-1]))
            self._send(200, "image/jpeg", d) if d else self.send_error(404)
        elif p.startswith("/api/framedata/"):
            si = int(p.split("/")[-1])
            fd = self.tracker.get_frame_data(si)
            self._json(fd) if fd else self.send_error(404)
        elif p.startswith("/api/mask/"):
            si = int(p.split("/")[-1])
            d = self.tracker.mask_png(si)
            self._send(200, "image/png", d) if d else self.send_error(404)
        elif p == "/api/bbox_ranges":
            self._json({"ranges": self.tracker.bbox_ranges()})
        elif p == "/api/progress":
            s = self.tracker.state
            self._json({"running": s.running, "phase": s.phase,
                "progress": s.progress, "total": s.total,
                "error": s.error, "start": s.start, "end": s.end,
                "objects": {str(k): v for k, v in s.objects.items()}})
        else:
            self.send_error(404)

    def do_POST(self):
        p = self.path.split("?")[0]
        body = self._body()
        if p == "/api/track":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.start_tracking(self.video, int(d["frame"]),
                d["boxes"], int(d.get("duration", TRACK_SECONDS)))
            self._json({"ok": True})
        elif p == "/api/extend_track":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.extend_object(self.video, int(d["oid"]),
                int(d.get("duration", TRACK_SECONDS)))
            self._json({"ok": True})
        elif p == "/api/update_bbox":
            d = json.loads(body)
            ok = self.tracker.update_bbox(int(d["sample"]), int(d["oid"]), d["bbox"])
            self._json({"ok": ok})
        elif p == "/api/relabel":
            d = json.loads(body)
            ok = self.tracker.relabel(int(d["oid"]), d["label"])
            self._json({"ok": ok})
        elif p == "/api/toggle_oof":
            d = json.loads(body)
            is_oof = self.tracker.toggle_oof(int(d["sample"]), int(d["oid"]))
            self._json({"ok": True, "is_oof": is_oof})
        elif p == "/api/update_nblobs":
            d = json.loads(body)
            ok = self.tracker.update_nblobs(int(d["oid"]), int(d["n_blobs"]))
            self._json({"ok": ok})
        elif p == "/api/set_frame":
            d = json.loads(body)
            self.tracker.state.current_frame = int(d.get("frame", 0))
            self._json({"ok": True})
        elif p == "/api/delete_object":
            d = json.loads(body)
            ok = self.tracker.delete_object(int(d["oid"]))
            self._json({"ok": ok})
        elif p == "/api/reset":
            self.tracker.reset()
            self._json({"ok": True})
        elif p == "/api/export":
            d = json.loads(body)
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            et = d.get("type", "video")
            try:
                if et == "video":
                    out = str(EXPORT_DIR / "tracked.mp4")
                    self._json({"ok": self.tracker.export_video(self.video, out), "path": out})
                elif et == "data":
                    out = str(EXPORT_DIR / "masks.npz")
                    self._json({"ok": self.tracker.export_data(out), "path": str(EXPORT_DIR)})
                else:
                    self._json({"ok": False, "error": "unknown type"})
            except Exception as e:
                self._json({"ok": False, "error": str(e)})
        else:
            self.send_error(404)

    def _send(self, code, ct, body):
        self.send_response(code); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""
    def log_message(self, *a): pass

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    video = VideoReader(VIDEO_PATH, sample_fps=SAMPLE_FPS)

    tracker = SAM2Tracker()
    if not tracker.available:
        print("\n  ⚠  SAM2 NOT FOUND")
        print("  Install:  pip install sam2   # or:")
        print("            git clone https://github.com/facebookresearch/sam2.git ~/sam2")
        print("            cd ~/sam2 && pip install -e .\n")

    # Autoload previous session
    if tracker._autoload():
        print("[main] Restored previous tracking session")

    if "--export-labeled" in sys.argv:
        out_path = EXPORT_DIR / "ALTR-20.mp4"
        ok = tracker.export_video(video, str(out_path))
        print(f"[export] {'OK' if ok else 'FAIL'} → {out_path}")
        sys.exit(0)

    Handler.video = video
    Handler.tracker = tracker
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n{'='*58}")
    print(f"   http://localhost:{PORT}")
    print(f"   ssh -L {PORT}:localhost:{PORT} user@host")
    print(f"{'='*58}")
    print(f"\n   {video.total} samples @ {SAMPLE_FPS} fps — SAM2 box prompts")
    print(f"   Autosave: {AUTOSAVE_DIR}\n")
    print("  Features:")
    print("  • Draw boxes + set blob count → track with SAM2")
    print("  • Show masks AND bounding boxes (separate toggles)")
    print("  • Edit tracked bboxes (drag/resize)")
    print("  • Change instrument class via dropdown")
    print("  • Mark objects out-of-frame per frame")
    print("  • Delete entire object series")
    print("  • Autosaves — restart to continue where you left off\n")

    try: server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving…"); tracker._autosave()
        print("Done."); video.close(); server.server_close()
