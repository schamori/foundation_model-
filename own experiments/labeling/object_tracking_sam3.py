#!/usr/bin/env python3
"""
Surgical Instrument Video Tracker — SAM3 + Editable Bounding Boxes

Features:
  - SAM3 text prompts for each instrument label
  - Masks → bboxes via largest N connected components (configurable blobs)
  - Show both masks AND bounding boxes (separate toggles)
  - Editable bboxes (drag move/resize) per frame
  - Change object class/label after tracking
  - Mark object "out of frame" per frame
  - Delete entire object series
  - Autosave/autoload — resume where you left off
  - Multiple tracking runs merge (no reset needed)
  - Always-interactive (no mode lock)

Install SAM3:
  git clone https://github.com/facebookresearch/sam3.git ~/sam3
  cd ~/sam3 && pip install -e .
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
VIDEO_PATH    = Path("/home/moritz/Foundential Model/own experiments/labeling/test_videos/ATLR_3.mp4")
TRIMMED_PATH  = VIDEO_PATH.parent / (VIDEO_PATH.stem + "_trimmed.mp4")
START_TIME    = "00:00:00"
PORT          = 8765
SAMPLE_FPS    = 1
TRACK_SECONDS = 120
EXPORT_DIR    = Path("./tracking_exports")
AUTOSAVE_DIR  = EXPORT_DIR / "autosave"

INSTRUMENTS = [
    {"name": "Bipolar",         "hex": "#ef476f", "rgb": [239,  71, 111]},
    {"name": "Microdissectors", "hex": "#06d6a0", "rgb": [  6, 214, 160]},
    {"name": "Suction",         "hex": "#118ab2", "rgb": [ 17, 138, 178]},
    {"name": "Drills",          "hex": "#ffd166", "rgb": [255, 209, 102]},
    {"name": "Microscissors",   "hex": "#9b59b6", "rgb": [155,  89, 182]},
    {"name": "CUSA",            "hex": "#e67e22", "rgb": [230, 126,  34]},
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

# ═══════════════════════════════════════════════════════════════
#  TRIM VIDEO
# ═══════════════════════════════════════════════════════════════
def trim_video():
    if TRIMMED_PATH.exists():
        print(f"[trim] Exists: {TRIMMED_PATH}"); return
    print(f"[trim] Trimming from {START_TIME} …")
    subprocess.run([
        "ffmpeg","-y","-ss",START_TIME,"-i",str(VIDEO_PATH),
        "-c","copy","-avoid_negative_ts","make_zero",str(TRIMMED_PATH)
    ], check=True)

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
#  SAM3 TRACKER
# ═══════════════════════════════════════════════════════════════
def _load_predictor():
    try:
        from sam3.model_builder import build_sam3_video_predictor
        print("[sam3] Building predictor …")
        p = build_sam3_video_predictor()
        print("[sam3] Loaded ✓")
        return p
    except Exception as e:
        print(f"[sam3] Failed: {e}")
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
        self.oof       = set()  # {(sample_idx, oid)} — out-of-frame markers
        self.start = self.end = None
        self.running = False
        self.phase = ""
        self.progress = self.total = 0
        self.error = None


class SAM3Tracker:
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
        s = self.state
        key = (si, oid)
        if key in s.oof:
            s.oof.discard(key)
            is_oof = False
        else:
            s.oof.add(key)
            is_oof = True
        self._autosave_debounced()
        return is_oof

    def delete_object(self, oid):
        s = self.state
        if oid not in s.objects: return False
        del s.objects[oid]
        for si in list(s.masks.keys()):
            s.masks[si].pop(oid, None)
            s.bboxes.get(si, {}).pop(oid, None)
        s.oof = {(si,o) for si,o in s.oof if o != oid}
        self._autosave()
        return True

    # ─── queries ──────────────────────────────────
    def get_frame_data(self, si):
        """Return all object data for a frame (bboxes, oof status)."""
        s = self.state
        result = {}
        bb = s.bboxes.get(si, {})
        for oid, info in s.objects.items():
            is_oof = (si, oid) in s.oof
            box = bb.get(oid)
            if is_oof:
                box = None
            if box is not None or is_oof:
                rgb = list(INST_COLOR.get(info["label"], (255,255,255)))
                result[str(oid)] = {
                    "label": info["label"], "bbox": box,
                    "color": rgb, "oof": is_oof,
                    "n_blobs": info.get("n_blobs", 1),
                }
        return result if result else None

    def mask_png(self, si):
        """Return RGBA PNG overlay of masks for a frame."""
        s = self.state
        om = s.masks.get(si)
        if not om: return None
        first = next(iter(om.values()))
        h, w = first.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for oid, mask in om.items():
            if (si, oid) in s.oof: continue
            info = s.objects.get(oid, {})
            r, g, b = INST_COLOR.get(info.get("label",""), (255,255,255))
            rgba[mask, 0] = b  # BGRA
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
            bb = s.bboxes.get(si, {})
            # draw mask overlay
            om = s.masks.get(si, {})
            overlay = frame.copy()
            for oid, mask in om.items():
                if (si, oid) in s.oof: continue
                info = s.objects.get(oid, {})
                c = np.array(INST_COLOR.get(info.get("label",""), (255,255,255))[::-1], dtype=np.float64)
                overlay[mask] = (overlay[mask]*0.6 + c*0.4).astype(np.uint8)
            frame = overlay
            # draw bboxes
            for oid, box in bb.items():
                if box is None or (si, oid) in s.oof: continue
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
        # masks npz
        arrays = {}
        for si, om in sorted(s.masks.items()):
            for oid, mask in om.items():
                lbl = s.objects.get(oid, {}).get("label", "unk")
                arrays[f"s{si:06d}_{lbl}_o{oid}"] = mask
        if arrays:
            np.savez_compressed(path, **arrays)
        # bboxes json
        bbox_path = str(path).replace(".npz", "_bboxes.json")
        bbox_export = {}
        for si in sorted(s.bboxes.keys()):
            fb = {}
            for oid, box in s.bboxes[si].items():
                is_oof = (si, oid) in s.oof
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
            # state json (bboxes + objects + oof)
            bboxes_ser = {}
            for si, bb in s.bboxes.items():
                bboxes_ser[str(si)] = {str(oid): box for oid, box in bb.items()}
            oof_ser = [[si, oid] for si, oid in s.oof]
            data = {
                "objects": {str(k): v for k, v in s.objects.items()},
                "bboxes": bboxes_ser,
                "oof": oof_ser,
                "start": s.start, "end": s.end,
            }
            with open(AUTOSAVE_DIR / "state.json", "w") as f:
                json.dump(data, f)
            # masks npz
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
            for si_str, bb in data.get("bboxes", {}).items():
                si = int(si_str)
                s.bboxes[si] = {}
                for oid_str, box in bb.items():
                    s.bboxes[si][int(oid_str)] = box
            s.oof = {(si, oid) for si, oid in data.get("oof", [])}
            # load masks
            mp = AUTOSAVE_DIR / "masks.npz"
            if mp.exists():
                npz = np.load(mp)
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

    # ─── SAM3 tracking run ────────────────────────
    def _run(self, video, sample_idx, boxes, duration_sec, start_oid):
        s = self.state
        session_id = None
        try:
            n_frames = min(duration_sec, video.total - sample_idx)
            s.total = n_frames
            new_start, new_end = sample_idx, sample_idx + n_frames - 1
            s.start = min(s.start, new_start) if s.start is not None else new_start
            s.end   = max(s.end, new_end) if s.end is not None else new_end

            # Phase 1 — extract frames
            s.phase = "extracting"
            self._tmp = tempfile.mkdtemp(prefix="sam3_")
            written = video.extract_range_to_dir(sample_idx, n_frames, self._tmp)
            print(f"[sam3] Extracted {written} frames → {self._tmp}")

            # Phase 2 — start session
            s.phase = "embedding"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):

                resp = self.predictor.handle_request(
                    request={"type": "start_session", "resource_path": self._tmp}
                )
            session_id = resp["session_id"]
            print(f"[sam3] Session: {session_id}")

            # Phase 3 — add prompts (text = instrument label)
            n_blobs_map = {}
            for i, binfo in enumerate(boxes):
                oid = start_oid + i
                label = f"We want to detect instruments in a surgical scence. Dont mask the background but just the instruments. Here is the label we are spefically looking for: {binfo['label']}"
                n_blobs_map[oid] = int(binfo.get("n_blobs", 1))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                    resp = self.predictor.handle_request(request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "obj_id": oid,
                        "text": label,
                    })
                print(f"[sam3] Prompt oid={oid} text='{label}'")
                # Log output structure on first prompt for debugging
                outs = resp.get("outputs")
                if outs is not None and i == 0:
                    print(f"[sam3]   add_prompt outputs: type={type(outs).__name__}"
                          f"{' keys='+str(list(outs.keys())[:5]) if isinstance(outs, dict) else ''}")

            # Phase 4 — propagate via handle_stream_request (generator)
            s.phase = "tracking"
            logged_first = False
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                for result in self.predictor.handle_stream_request(request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "max_frame_num_to_track": n_frames,
                }):
                    frame_idx = result["frame_index"]
                    outputs = result["outputs"]
                    abs_si = sample_idx + frame_idx
                    if abs_si not in s.masks:
                        s.masks[abs_si] = {}
                        s.bboxes[abs_si] = {}
                    # Log first frame's output structure for debugging
                    if not logged_first:
                        print(f"[sam3]   propagate outputs: type={type(outputs).__name__}"
                            f"{' keys='+str(list(outputs.keys())[:5]) if isinstance(outputs, dict) else ''}")
                        if isinstance(outputs, dict):
                            for k, v in list(outputs.items())[:3]:
                                vinfo = f"type={type(v).__name__}"
                                if hasattr(v, 'shape'): vinfo += f" shape={v.shape}"
                                elif hasattr(v, '__len__'): vinfo += f" len={len(v)}"
                                print(f"[sam3]     {k}: {vinfo}")
                        logged_first = True
                    self._parse_outputs(abs_si, outputs, n_blobs_map)
                    s.progress = frame_idx + 1

            # Close session
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                    self.predictor.handle_request(
                        request={"type": "close_session", "session_id": session_id}
                    )
            except: pass
            session_id = None  # prevent double-close in finally

            n_total = sum(1 for bb in s.bboxes.values() for b in bb.values() if b is not None)
            print(f"[sam3] Done — {len(s.masks)} frames, {len(s.objects)} objects, {n_total} bboxes ✓")
            self._autosave()

        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            s.running = False
            if session_id:
                try:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                        self.predictor.handle_request(
                            request={"type": "close_session", "session_id": session_id})
                except: pass
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None

    @staticmethod
    def _to_bool(m):
        if hasattr(m, 'cpu'): m = m.cpu().numpy()
        m = np.squeeze(m)
        return m > 0.0 if m.dtype != bool else m

    def _parse_outputs(self, abs_si, outputs, n_blobs_map):
        """Parse per-frame outputs from SAM3 propagation.

        SAM3 outputs format:
          {
            'out_obj_ids': ndarray of int obj IDs,        shape (N,)
            'out_binary_masks': tensor/ndarray of masks,  shape (N, 1, H, W) or (N, H, W)
            'out_boxes_xywh': ndarray of boxes,           shape (N, 4) in xywh
            'out_probs': ndarray of confidence scores,    shape (N,)
            'frame_stats': dict of frame-level stats
          }
        N can be 0 if no objects detected on this frame.
        """
        s = self.state
        if abs_si not in s.masks:
            s.masks[abs_si] = {}
            s.bboxes[abs_si] = {}

        if not isinstance(outputs, dict):
            return

        obj_ids = outputs.get("out_obj_ids")
        masks = outputs.get("out_binary_masks")

        if obj_ids is None or masks is None:
            return

        # Convert to numpy if needed
        if hasattr(obj_ids, 'cpu'):
            obj_ids = obj_ids.cpu().numpy()
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()

        obj_ids = np.asarray(obj_ids).flatten()

        # No detections on this frame
        if len(obj_ids) == 0:
            return

        for i, raw_oid in enumerate(obj_ids):
            oid = int(raw_oid)
            if i >= len(masks):
                break
            mask = self._to_bool(masks[i])
            if mask.ndim >= 2:
                nb = n_blobs_map.get(oid, s.objects.get(oid, {}).get("n_blobs", 1))
                s.masks[abs_si][oid] = mask
                s.bboxes[abs_si][oid] = mask_to_bbox(mask, nb)

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
#canvas{display:block;max-width:100%;cursor:crosshair}
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
.tobj:hover{background:#1a2535}
.tobj select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:11px;font-family:'JetBrains Mono',monospace;cursor:pointer}
.tobj .oof-btn{font-size:10px;padding:2px 6px;border:1px solid var(--border);border-radius:3px;cursor:pointer;background:var(--bg);color:var(--dim);transition:all .15s}
.tobj .oof-btn.active{background:#3a1c24;color:var(--red);border-color:var(--red)}
.tobj .oof-btn:hover{border-color:var(--border-h)}
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
.tl-tr{position:absolute;top:0;height:100%;background:rgba(0,180,255,.15);border-left:1px solid var(--accent);border-right:1px solid var(--accent)}
.tl-cur{position:absolute;top:0;width:2px;height:100%;background:var(--red);pointer-events:none}
.hidden{display:none!important}
.toast{position:fixed;bottom:20px;right:20px;background:#1a2636;border:1px solid var(--border);padding:10px 18px;border-radius:8px;font-size:13px;color:var(--green);transform:translateY(80px);opacity:0;transition:all .3s;z-index:99}
.toast.show{transform:translateY(0);opacity:1}
.toast.err{color:var(--red)}
</style></head>
<body>
<div class="app">
  <div class="header">
    <h1>▸ INSTRUMENT TRACKER</h1>
    <span class="pill pill-sam" id="samPill">SAM3: …</span>
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
    </div>
  </div>
  <div class="actions">
    <button class="btn-danger" onclick="clearDrawn()">clear drawn</button>
    <button class="btn-track" id="trackBtn" onclick="startTrack()" disabled>▸ track</button>
    <label>dur <input type="number" id="durIn" value="120" min="5" max="3600">s</label>
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
    <div class="tl-tr hidden" id="tlTr"></div>
    <div class="tl-cur" id="tlCur" style="left:0"></div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
var S={frame:0,total:0,vw:0,vh:0,boxes:[],activeLabel:'',activeColor:'',
  instruments:[],tracked:null,fdCache:{},maskCache:{},tracking:false,playTimer:null,
  allObjects:{}};
var canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
var frameImg=null,maskImg=null,drag=null;

/* ── init ─────────────────────────────── */
fetch('/api/info').then(r=>r.json()).then(d=>{
  S.total=d.total_samples;S.vw=d.width;S.vh=d.height;S.instruments=d.instruments;
  document.getElementById('iTotal').textContent=S.total;
  document.getElementById('iRes').textContent=d.width+'×'+d.height;
  document.getElementById('fpsPill').textContent=d.sample_fps+' FPS';
  var sp=document.getElementById('samPill');
  sp.textContent=d.sam3_ok?'SAM3: ready':'SAM3: not found';
  sp.classList.toggle('off',!d.sam3_ok);
  var sel=document.getElementById('labelSel');
  d.instruments.forEach((inst,i)=>{
    var b=document.createElement('button');
    b.className='label-pill'+(i===0?' active':'');
    b.dataset.label=inst.name;b.style.setProperty('--clr',inst.hex);b.style.color=inst.hex;
    b.textContent=inst.name;b.onclick=()=>selLbl(inst.name,inst.hex);sel.appendChild(b);
  });
  if(d.instruments.length){S.activeLabel=d.instruments[0].name;S.activeColor=d.instruments[0].hex;}
  /* check for restored tracked data */
  if(d.has_tracked){
    S.tracked={start:d.tracked_start,end:d.tracked_end};
    S.allObjects=d.objects||{};
    updTimeline();updUI();
  }
  loadFrame(0);
});

function selLbl(n,h){S.activeLabel=n;S.activeColor=h;
  document.querySelectorAll('.label-pill').forEach(b=>b.classList.toggle('active',b.dataset.label===n));}
function instHex(l){var i=S.instruments.find(x=>x.name===l);return i?i.hex:'#fff';}
function rgb(a){return 'rgb('+a[0]+','+a[1]+','+a[2]+')';}

/* ── frame loading ────────────────────── */
function loadImg(u){return new Promise((r,j)=>{var i=new Image();i.onload=()=>r(i);i.onerror=j;i.src=u;});}
async function loadFrame(n){
  n=Math.max(0,Math.min(n,S.total-1));S.frame=n;
  document.getElementById('fIn').value=n;
  document.getElementById('iFrame').textContent=n;
  document.getElementById('iTime').textContent=n+'s';
  document.getElementById('tlCur').style.left=(S.total>1?(n/(S.total-1))*100:0)+'%';
  try{frameImg=await loadImg('/api/frame/'+n+'?t='+Date.now());}catch(e){return;}
  canvas.width=frameImg.naturalWidth;canvas.height=frameImg.naturalHeight;
  maskImg=null;
  if(S.tracked&&n>=S.tracked.start&&n<=S.tracked.end){
    if(S.fdCache[n]===undefined){
      try{var r=await fetch('/api/framedata/'+n);S.fdCache[n]=r.ok?await r.json():null;}
      catch(e){S.fdCache[n]=null;}}
    if(S.maskCache[n]===undefined){
      try{maskImg=await loadImg('/api/mask/'+n);S.maskCache[n]=maskImg;}
      catch(e){S.maskCache[n]=null;}}
    else maskImg=S.maskCache[n];
  }
  redr();updTrkList();
}

/* ── hit testing ──────────────────────── */
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
    if(x>x1&&x<x2&&y>y1&&y<y2)return{type:'move',oid,cursor:'move'};
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

/* ── mouse ────────────────────────────── */
canvas.addEventListener('mousedown',e=>{
  if(S.tracking)return;var p=cCoords(e),hit=hitTest(p.x,p.y);
  if(hit){var bb=S.fdCache[S.frame][hit.oid].bbox;
    drag={type:hit.type,oid:hit.oid,handle:hit.handle,sx:p.x,sy:p.y,orig:bb.slice(),box:bb.slice()};}
  else drag={type:'draw',sx:p.x,sy:p.y,box:null};
});
canvas.addEventListener('mousemove',e=>{
  var p=cCoords(e);
  if(!drag){var h=hitTest(p.x,p.y);canvas.style.cursor=h?h.cursor:'crosshair';return;}
  if(drag.type==='draw')drag.box=[Math.min(drag.sx,p.x),Math.min(drag.sy,p.y),Math.max(drag.sx,p.x),Math.max(drag.sy,p.y)];
  else drag.box=dragBox(p.x,p.y);
  redr();
});
canvas.addEventListener('mouseup',e=>{
  if(!drag)return;
  if(drag.type==='draw'){var b=drag.box;
    if(b&&(b[2]-b[0]>10)&&(b[3]-b[1]>10)){
      S.boxes.push({x1:b[0],y1:b[1],x2:b[2],y2:b[3],label:S.activeLabel,color:S.activeColor,n_blobs:1});
      updBoxList();}}
  else if(drag.box){S.fdCache[S.frame][drag.oid].bbox=drag.box;
    fetch('/api/update_bbox',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sample:S.frame,oid:parseInt(drag.oid),bbox:drag.box.map(v=>Math.round(v))})});}
  drag=null;updUI();redr();
});
canvas.addEventListener('mouseleave',()=>{if(drag&&drag.type!=='draw'){drag=null;redr();}});

/* ── drawing ──────────────────────────── */
function redr(){
  if(!frameImg)return;ctx.drawImage(frameImg,0,0);
  /* masks */
  if(maskImg&&document.getElementById('cbMask').checked){
    ctx.globalAlpha=0.5;ctx.drawImage(maskImg,0,0);ctx.globalAlpha=1;}
  /* tracked bboxes */
  if(document.getElementById('cbBox').checked){
    var fd=S.fdCache[S.frame];
    if(fd){for(var oid in fd){var o=fd[oid];if(!o||!o.bbox||o.oof)continue;
      var box=o.bbox;if(drag&&drag.oid===oid&&drag.box)box=drag.box;
      drawTBox(box[0],box[1],box[2],box[3],rgb(o.color),o.label,o.color);}}}
  /* new drawn boxes */
  S.boxes.forEach(b=>drawNBox(b.x1,b.y1,b.x2,b.y2,b.color,b.label));
  if(drag&&drag.type==='draw'&&drag.box){var db=drag.box;drawNBox(db[0],db[1],db[2],db[3],S.activeColor,S.activeLabel);}
}
function drawTBox(x1,y1,x2,y2,col,lbl,ca){
  ctx.strokeStyle=col;ctx.lineWidth=3;ctx.setLineDash([]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.fillStyle='rgba('+ca[0]+','+ca[1]+','+ca[2]+',0.08)';ctx.fillRect(x1,y1,x2-x1,y2-y1);
  ctx.font='bold 13px IBM Plex Sans,sans-serif';var tw=ctx.measureText(lbl).width+10,th=20;
  var ly=y1>th+4?y1-th-2:y2+2;ctx.fillStyle=col;ctx.fillRect(x1,ly,tw,th);
  ctx.fillStyle='#fff';ctx.fillText(lbl,x1+5,ly+15);
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

/* ── box list (new) ───────────────────── */
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

/* ── tracked objects list (per-frame) ──── */
function updTrkList(){
  var el=document.getElementById('trkList');
  var fd=S.fdCache[S.frame];
  var objs=S.allObjects;
  if(!objs||!Object.keys(objs).length){el.innerHTML='<span class="bempty">No tracked objects</span>';return;}
  var h='';
  for(var oid in objs){
    var o=objs[oid],c=instHex(o.label);
    var isOof=fd&&fd[oid]&&fd[oid].oof;
    var hasBbox=fd&&fd[oid]&&fd[oid].bbox;
    h+='<div class="tobj"><span class="bdot" style="background:'+c+'"></span>';
    h+='<select onchange="relabel('+oid+',this.value)">';
    S.instruments.forEach(inst=>{
      h+='<option value="'+inst.name+'"'+(inst.name===o.label?' selected':'')+'>'+inst.name+'</option>';});
    h+='</select>';
    h+='<span class="oof-btn'+(isOof?' active':'')+'" onclick="togOof('+oid+')">'+(isOof?'OOF':'vis')+'</span>';
    h+='<button onclick="delObj('+oid+')" title="Delete entire series">×</button></div>';
  }
  el.innerHTML=h;
}

/* ── UI visibility ────────────────────── */
function updUI(){
  var hb=S.boxes.length>0,ht=S.tracked!==null;
  document.getElementById('trackBtn').disabled=!hb||S.tracking;
  ['cbMaskLbl','cbBoxLbl','expVid','expDat','rstBtn'].forEach(id=>
    document.getElementById(id).classList.toggle('hidden',!ht));
  document.getElementById('trkWrap').classList.toggle('hidden',!ht);
  document.getElementById('progRow').classList.toggle('hidden',!S.tracking);
}

/* ── navigation ───────────────────────── */
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

/* ── tracking ─────────────────────────── */
async function startTrack(){
  if(!S.boxes.length||S.tracking)return;
  var dur=+document.getElementById('durIn').value||120;
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
    if(d.phase==='tracking')t='tracking '+d.progress+'/'+d.total;
    else if(d.phase==='embedding')t='computing embeddings…';
    else if(d.phase==='extracting')t='extracting frames…';
    document.getElementById('pTxt').textContent=t;
    if(d.error){toast(d.error,true);S.tracking=false;updUI();return;}
    if(d.running){setTimeout(pollProg,400);return;}
    S.tracking=false;S.tracked={start:d.start,end:d.end};
    S.fdCache={};S.maskCache={};S.boxes=[];S.allObjects=d.objects||{};
    updBoxList();updTimeline();updUI();
    toast('Tracked '+(d.end-d.start+1)+'s — '+Object.keys(d.objects).length+' objects');
    loadFrame(S.frame);
  }catch(e){setTimeout(pollProg,1000);}
}
function updTimeline(){var el=document.getElementById('tlTr');
  if(!S.tracked){el.classList.add('hidden');return;}el.classList.remove('hidden');
  el.style.left=(S.tracked.start/Math.max(S.total-1,1))*100+'%';
  el.style.width=((S.tracked.end-S.tracked.start)/Math.max(S.total-1,1))*100+'%';}

/* ── object actions ───────────────────── */
async function relabel(oid,newLabel){
  await fetch('/api/relabel',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,label:newLabel})});
  S.allObjects[oid].label=newLabel;S.fdCache={};S.maskCache={};loadFrame(S.frame);
  toast('Relabeled obj '+oid+' → '+newLabel);
}
async function togOof(oid){
  var r=await fetch('/api/toggle_oof',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sample:S.frame,oid:oid})});
  var d=await r.json();
  if(S.fdCache[S.frame]&&S.fdCache[S.frame][oid])S.fdCache[S.frame][oid].oof=d.is_oof;
  else{delete S.fdCache[S.frame];}
  S.maskCache={};loadFrame(S.frame);
}
async function delObj(oid){
  if(!confirm('Delete object '+oid+' from ALL frames?'))return;
  await fetch('/api/delete_object',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid})});
  delete S.allObjects[String(oid)];S.fdCache={};S.maskCache={};loadFrame(S.frame);updUI();
  toast('Deleted obj '+oid);
}
function rstAll(){
  if(!confirm('Reset ALL tracking data?'))return;
  fetch('/api/reset',{method:'POST'});
  S.tracked=null;S.fdCache={};S.maskCache={};S.boxes=[];S.allObjects={};
  updBoxList();updTimeline();updUI();document.getElementById('trkList').innerHTML='';redr();
  toast('Reset');
}

/* ── export ───────────────────────────── */
async function doExp(t){toast('Exporting…');
  try{var r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({type:t})});var d=await r.json();
    d.ok?toast('Exported → '+d.path):toast(d.error||'fail',true);}catch(e){toast('fail',true);}}

/* ── toast ────────────────────────────── */
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
    tracker: SAM3Tracker  = None

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
                "sam3_ok": self.tracker.available,
                "instruments": INSTRUMENTS,
                "has_tracked": has_tracked,
                "tracked_start": s.start,
                "tracked_end": s.end,
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
                self._json({"ok": False, "error": "SAM3 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.start_tracking(self.video, int(d["frame"]),
                d["boxes"], int(d.get("duration", TRACK_SECONDS)))
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
    trim_video()
    video = VideoReader(TRIMMED_PATH, sample_fps=SAMPLE_FPS)

    tracker = SAM3Tracker()
    if not tracker.available:
        print("\n  ⚠  SAM3 NOT FOUND")
        print("  Install:  git clone https://github.com/facebookresearch/sam3.git ~/sam3")
        print("            cd ~/sam3 && pip install -e .\n")

    # Autoload previous session
    if tracker._autoload():
        print("[main] Restored previous tracking session")

    Handler.video = video
    Handler.tracker = tracker
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n{'='*58}")
    print(f"   http://localhost:{PORT}")
    print(f"   ssh -L {PORT}:localhost:{PORT} user@host")
    print(f"{'='*58}")
    print(f"\n   {video.total} samples @ {SAMPLE_FPS} fps — SAM3 text prompts")
    print(f"   Autosave: {AUTOSAVE_DIR}\n")
    print("  Features:")
    print("  • Draw boxes + set blob count → SAM3 tracks with text labels")
    print("  • Edit tracked bboxes (drag/resize)")
    print("  • Change instrument class via dropdown")
    print("  • Mark objects out-of-frame per frame")
    print("  • Delete entire object series")
    print("  • Show masks AND bounding boxes (separate toggles)")
    print("  • Autosaves — restart to continue where you left off\n")

    try: server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving…"); tracker._autosave()
        print("Done."); video.close(); server.server_close()