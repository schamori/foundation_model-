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

import os, sys, json, re, shutil, subprocess, threading, traceback, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
import torch

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
VIDEO_PATH    = None   # set automatically from last-used video; use the picker to select
PORT          = 8766
SAMPLE_FPS    = 1
TRACK_SECONDS = 50
SAM2_HF_ID   = "facebook/sam2-hiera-large"
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
EXPORT_DIR    = PROJECT_ROOT / "tracking_exports"
AUTOSAVE_DIR  = None   # computed once a video is loaded

# Temporal filtering.
# Set FEATURES_ROOT to a Path to pin a specific directory, or leave None to
# auto-discover (searches sibling directories of VIDEO_PATH and known paths).
# Expected layout: FEATURES_ROOT / VIDEO_PATH.stem / *.npy
FILTERED_FRAMES_ROOT  = Path("/media/HDD1/moritz/Extracted Frames")
EXTRACTED_FRAMES_ROOT = Path("/media/HDD1/moritz/Extracted Frames")
FEATURES_ROOT         = None   # auto-discovered when None
VIDEO_PICKER_DIR      = Path("/media/HDD1/moritz/foundential/test")   # directory with .mp4 files shown in the picker
CLIP_LENGTH           = 500   # frames per clip (aim for ~20 chunks across the video)
TEMPORAL_TOP_FRACTION = 0.20   # fraction of clips to select via diversity sampling
TEMPORAL_MAX_FRAME    = None    # ignore frames beyond this index (None = use all)

# Display / inference resolution.  All frames served to the frontend are
# downsampled to this size.  SAM2 masks are also produced at this resolution,
# so bounding-box coordinates always live in this pixel space.
# Set to None to use the native video resolution.
DISPLAY_RESOLUTION    = None   # (width, height)


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
        self._ffmpeg_bin = shutil.which("ffmpeg")
        print(f"[video] {self.orig_total} orig @ {self.orig_fps:.1f} fps → {self.total} samples (step {self.step})")
        if self._ffmpeg_bin:
            print(f"[video] ffmpeg extractor: {self._ffmpeg_bin}")
        else:
            print("[video] ffmpeg extractor: not found, using OpenCV fallback")

    def get_frame(self, sample_idx):
        if sample_idx < 0 or sample_idx >= self.total: return None
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.sample_map[sample_idx])
            ok, f = self._cap.read()
        return f if ok else None

    def frame_jpeg(self, sample_idx, q=85):
        f = self.get_frame(sample_idx)
        if f is None: return None
        if DISPLAY_RESOLUTION is not None:
            f = cv2.resize(f, DISPLAY_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes()

    def extract_range_to_dir(self, start_sample, count, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if count <= 0 or start_sample >= self.total:
            return 0
        start_sample = max(0, int(start_sample))
        wanted = min(int(count), self.total - start_sample)

        for p in Path(out_dir).glob("*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass

        if self._ffmpeg_bin:
            ff_written = self._extract_range_ffmpeg(start_sample, wanted, out_dir)
            if ff_written >= wanted:
                return ff_written
            if ff_written > 0:
                print(f"[video] ffmpeg partial ({ff_written}/{wanted}); filling remaining with OpenCV")
            else:
                print("[video] ffmpeg extraction failed; falling back to OpenCV")
            cv_written = self._extract_range_opencv(
                start_sample + ff_written, wanted - ff_written, out_dir, start_idx=ff_written
            )
            return ff_written + cv_written

        return self._extract_range_opencv(start_sample, wanted, out_dir, start_idx=0)

    def _extract_range_ffmpeg(self, start_sample, count, out_dir):
        if count <= 0:
            return 0
        start_orig = self.sample_map[start_sample]
        start_sec = start_orig / max(self.orig_fps, 1e-6)
        out_pattern = os.path.join(out_dir, "%06d.jpg")
        cmd = [
            self._ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start_sec:.6f}",
            "-i", self.path,
            "-an", "-sn",
            "-threads", "0",
            "-frames:v", str(count),
            "-start_number", "0",
            "-q:v", "2",
        ]
        if self.step > 1:
            cmd += ["-vf", f"select=not(mod(n\\,{self.step}))", "-vsync", "0"]
        cmd += [out_pattern]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                err = (proc.stderr or "").strip()
                print(f"[video] ffmpeg error: {err[:300]}")
                return 0
            return sum(1 for _ in Path(out_dir).glob("*.jpg"))
        except Exception as e:
            print(f"[video] ffmpeg exception: {e}")
            return 0

    def _extract_range_opencv(self, start_sample, count, out_dir, start_idx=0):
        written = 0
        for i in range(count):
            si = start_sample + i
            if si >= self.total:
                break
            f = self.get_frame(si)
            if f is None:
                break
            out_i = start_idx + i
            cv2.imwrite(os.path.join(out_dir, f"{out_i:06d}.jpg"), f)
            written += 1
        return written

    def close(self): self._cap.release()

# ═══════════════════════════════════════════════════════════════
#  FRAME DIRECTORY READER
# ═══════════════════════════════════════════════════════════════
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.npy'}

class FrameReader:
    """Read frames from a sorted directory of image files (same interface as VideoReader)."""
    def __init__(self, path, sample_fps=SAMPLE_FPS):
        self.path = Path(path)
        self._files = sorted(
            f for f in self.path.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        )
        if not self._files:
            raise ValueError(f"No image files found in {path}")
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise ValueError(f"Cannot read first image: {self._files[0]}")
        self.height, self.width = first.shape[:2]
        self.orig_fps    = sample_fps
        self.orig_total  = len(self._files)
        self.step        = 1        # every file is one sample
        self.sample_map  = list(range(self.orig_total))
        self.total       = self.orig_total
        self._lock       = threading.Lock()
        self._disk_count = self.orig_total   # last known file count on disk
        print(f"[frames] {self.total} images in '{self.path.name}'")

    def refresh(self) -> bool:
        """Re-scan directory; rebuild _files if count changed. Returns True if refreshed."""
        new_files = sorted(
            f for f in self.path.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        )
        if len(new_files) == self._disk_count:
            return False
        with self._lock:
            old = self._disk_count
            self._files = new_files
            self.orig_total = len(new_files)
            self.sample_map = list(range(self.orig_total))
            self.total = self.orig_total
            self._disk_count = self.orig_total
        print(f"[frames] Refreshed '{self.path.name}': {old} → {self.total} images")
        return True

    def get_frame(self, sample_idx):
        if sample_idx < 0 or sample_idx >= self.total: return None
        with self._lock:
            return cv2.imread(str(self._files[sample_idx]))

    def frame_jpeg(self, sample_idx, q=85):
        f = self.get_frame(sample_idx)
        if f is None: return None
        if DISPLAY_RESOLUTION is not None:
            f = cv2.resize(f, DISPLAY_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        return buf.tobytes()

    def extract_range_to_dir(self, start_sample, count, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if count <= 0 or start_sample >= self.total:
            return 0
        start_sample = max(0, int(start_sample))
        wanted = min(int(count), self.total - start_sample)
        for p in Path(out_dir).glob("*.jpg"):
            try: p.unlink()
            except OSError: pass
        written = 0
        for i in range(wanted):
            si = start_sample + i
            if si >= self.total: break
            f = self.get_frame(si)
            if f is None: break
            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), f)
            written += 1
        return written

    def close(self): pass


def make_video_source(path, sample_fps=SAMPLE_FPS):
    """Return a FrameReader for a directory or a VideoReader for a video file."""
    p = Path(path)
    if p.is_dir():
        return FrameReader(p, sample_fps)
    return VideoReader(p, sample_fps)


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL CLIP SELECTOR
# ═══════════════════════════════════════════════════════════════

# Primary known root (structure: root/dataset/video_name/*.npy)
_KNOWN_FEAT_ROOTS = [
    Path("/media/HDD1/moritz/Extracted Frames embeddings"),
    Path("/media/HDD1/moritz/foundential/Extracted Frames Features"),
    Path("/media/HDD1/moritz/foundential/Extracted Frames Embeddings"),
    # baseline.ipynb Cholec80 embeddings (sibling of labeling/)
    Path(__file__).resolve().parents[1] / "Cholec80" / "DINO_cholec_embeddings",
    Path(__file__).resolve().parents[1] / "Cholec80" / "Convenxt_cholec_embeddings",
]
_FEAT_KEYWORDS = {"feature", "features", "embed", "embeddings", "emb", "feat"}


_PREFIX_MATCH_PREFIXES = ("RS", "TL")  # only stems starting with these use prefix matching
_TRAILING_NUM_RE = re.compile(r'(\d+)$')

def _prefix_key(stem: str) -> str:
    """Return the short ID prefix used for fuzzy matching.
    Only applies when the stem starts with one of _PREFIX_MATCH_PREFIXES.
    'RS-034_vestibular_schwannoma_...' → 'RS-034',  'TL-052' → 'TL-052'.
    All other stems are returned unchanged (no prefix matching)."""
    if any(stem.startswith(p) for p in _PREFIX_MATCH_PREFIXES):
        return stem.split('_')[0]
    return stem


def _trailing_digits(stem: str) -> str | None:
    """Return the trailing digit sequence of stem, or None.
    'Aneurysm_21905025' → '21905025',  'frame_000042' → '000042'."""
    m = _TRAILING_NUM_RE.search(stem)
    return m.group(1) if m else None


def _match_stem(stem: str, candidates) -> str:
    """Return the best match for stem among a collection of names.

    Priority:
      1. Exact match
      2. Starts-with prefix key  (RS-034, TL-052, …)
      3. Trailing-digits exact   'Aneurysm_21905025' ↔ 'MCA_..._21905025'
      4. Digit-ID substring      'Aneurysm_21144934' ↔ 'MCA_..._21144934_Volume_3_Video_1'
         (only for digit sequences ≥ 4 chars, matched as whole numbers)
    """
    cands = list(candidates)
    if stem in cands:
        return stem
    # Priority 1.5: candidate starts with the full stem (e.g. "5ALA_003" → "5ALA_003_25062025_...")
    for c in sorted(cands):
        if c.startswith(stem + "_") or c.startswith(stem + "-"):
            return c
    prefix = _prefix_key(stem)
    for c in sorted(cands):
        if c.startswith(prefix):
            return c
    td = _trailing_digits(stem)
    if td and len(td) >= 4:
        # Priority 3: candidate also ends with same digits (≥4 digits to skip generic "003" etc.)
        for c in sorted(cands):
            if _trailing_digits(c) == td:
                return c
        # Priority 4: digit-ID appears anywhere in candidate (as a whole number)
            td_re = re.compile(r'(?<!\d)' + re.escape(td) + r'(?!\d)')
            for c in sorted(cands):
                if td_re.search(c):
                    return c
    return stem


def _resolve_stem_in_dir(stem: str, directory: Path) -> str:
    """Find the best-matching existing subdirectory name for stem under directory.
    Returns stem unchanged when no match is found (caller will create a new dir)."""
    if not directory.is_dir():
        return stem
    return _match_stem(stem, (e.name for e in directory.iterdir() if e.is_dir()))


def _load_allowed_frames(filtered_root: Path | None, video, video_name: str | None = None) -> list[int] | None:
    """
    Return sorted list of allowed sample indices based on which frames exist in
    filtered_root/video_name.  Returns None if no filtering is configured.

    For FrameReader: matches by filename stem against video._files[i].stem.
    For VideoReader: parses numeric stems directly as sample indices.

    Searches recursively under filtered_root if the direct path is not found,
    so setting filtered_root to a parent directory (e.g. "test") will still
    find "test/VS_Retrosigmoid/RS-034_...".
    """
    if filtered_root is None:
        return None
    if video_name is None:
        video_name = VIDEO_PATH.stem
    folder = filtered_root / _resolve_stem_in_dir(video_name, filtered_root)
    if not folder.is_dir():
        # Search recursively — try exact name, prefix, or digit-ID match
        prefix = _prefix_key(video_name)
        td = _trailing_digits(video_name)
        td_re = re.compile(r'(?<!\d)' + re.escape(td) + r'(?!\d)') if td and len(td) >= 4 else None
        matches = [
            m for m in filtered_root.rglob("*")
            if m.is_dir() and (
                m.name == video_name
                or m.name.startswith(video_name + "_") or m.name.startswith(video_name + "-")
                or m.name.startswith(prefix)
                or (_trailing_digits(m.name) == td if td and len(td) >= 4 else False)
                or (td_re.search(m.name) if td_re else False)
            )
        ]
        if matches:
            # prefer exact > startswith(stem) > prefix/digit-ID hit
            exact = [m for m in matches if m.name == video_name]
            stem_match = [m for m in matches if m.name.startswith(video_name + "_") or m.name.startswith(video_name + "-")]
            folder = (exact or stem_match or matches)[0]
            print(f"[filter] Found folder: {folder}")
        else:
            print(f"[filter] Folder not found for '{video_name}' (prefix '{prefix}') under {filtered_root}")
            return None

    present_stems = {p.stem for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS}
    if not present_stems:
        # Folder matched but contains no images directly —
        # search one level deeper (e.g. test/MVD/ → test/MVD/RS-034_.../)
        prefix = _prefix_key(video_name)
        td = _trailing_digits(video_name)
        td_re = re.compile(r'(?<!\d)' + re.escape(td) + r'(?!\d)') if td and len(td) >= 4 else None
        deeper = None
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue
            sub_td = _trailing_digits(sub.name)
            if (sub.name == video_name
                    or sub.name.startswith(video_name + "_") or sub.name.startswith(video_name + "-")
                    or sub.name.startswith(prefix)
                    or (td and len(td) >= 4 and sub_td == td)
                    or (td_re and td_re.search(sub.name))):
                stems = {p.stem for p in sub.iterdir() if p.suffix.lower() in _IMG_EXTS}
                if stems:
                    deeper = sub
                    present_stems = stems
                    break
        if deeper is None:
            print(f"[filter] No images found in {folder} or its subdirectories")
            return None
        folder = deeper
        print(f"[filter] Found images one level deeper: {folder}")

    allowed = []
    if hasattr(video, "_files"):  # FrameReader — match by stem
        for i, f in enumerate(video._files):
            if f.stem in present_stems:
                allowed.append(i)
    else:  # VideoReader — extract trailing number from stem (handles "frame_000001" etc.)
        _num_re = re.compile(r'(\d+)$')
        for stem in present_stems:
            m = _num_re.search(stem)
            if m:
                allowed.append(int(m.group(1)))

    allowed.sort()
    total = video.total
    filtered_out = total - len(allowed)
    print(f"[filter] {len(allowed)}/{total} frames allowed ({filtered_out} filtered out)")
    return allowed


def _apply_custom_exclusions(allowed: list[int] | None, autosave_dir: Path | None) -> list[int] | None:
    """Remove frames listed in autosave_dir/custom.txt from allowed list.

    custom.txt format: one exclusion range per line as "start end" (inclusive).
    Example:
        1 5
        10 50
    This excludes sample indices 1-5 and 10-50.
    """
    if autosave_dir is None:
        return allowed
    custom = autosave_dir / "custom.txt"
    if not custom.exists():
        return allowed
    excluded = set()
    try:
        for line in custom.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                a, b = int(parts[0]), int(parts[1])
                excluded.update(range(a, b + 1))
    except Exception as e:
        print(f"[custom] Error reading {custom}: {e}")
        return allowed
    if not excluded:
        return allowed
    if allowed is None:
        print(f"[custom] Excluding {len(excluded)} frames (no base filter — cannot apply)")
        return allowed
    before = len(allowed)
    allowed = [f for f in allowed if f not in excluded]
    print(f"[custom] Excluded {before - len(allowed)} frames via custom.txt ({len(allowed)} remaining)")
    return allowed


def _find_features_root(video_path: Path) -> Path | None:
    """
    Auto-discover the features root containing per-frame .npy embeddings.

    Supports both flat   (root/video_name/*.npy)
    and nested           (root/dataset/video_name/*.npy)  layouts.

    Search order:
      1. Known hardcoded roots.
      2. Keyword-matching siblings of each ancestor directory of video_path.
    """
    video_name = video_path.stem

    def _has_embeddings(root: Path) -> bool:
        if not root.is_dir():
            return False
        # flat layout: root / {video_name or prefix-match}
        resolved = _resolve_stem_in_dir(video_name, root)
        direct = root / resolved
        if direct.is_dir() and any(direct.rglob("*.npy")):
            return True
        # nested layout: root / dataset / {video_name or prefix-match}
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            resolved_sub = _resolve_stem_in_dir(video_name, dataset_dir)
            sub = dataset_dir / resolved_sub
            if sub.is_dir() and any(sub.rglob("*.npy")):
                return True
        return False

    candidates: list[Path] = list(_KNOWN_FEAT_ROOTS)

    p = video_path if video_path.is_dir() else video_path.parent
    for _ in range(6):
        parent = p.parent
        try:
            for sibling in sorted(parent.iterdir()):
                if sibling.is_dir() and sibling != p and sibling not in candidates:
                    if any(kw in sibling.name.lower() for kw in _FEAT_KEYWORDS):
                        candidates.append(sibling)
        except PermissionError:
            pass
        p = parent

    for cand in candidates:
        if cand.exists() and _has_embeddings(cand):
            print(f"[temporal] Found features root: {cand}")
            return cand

    print(f"[temporal] No features root found for '{video_name}' — temporal filtering disabled")
    return None


def _load_video_embeddings(features_root: Path, video_name: str):
    """Load all .npy embeddings for a video folder, sorted by filename.

    Supports flat (features_root/video_name/) and nested
    (features_root/dataset/video_name/) layouts.
    """
    # flat layout — exact or prefix match
    vdir = features_root / _resolve_stem_in_dir(video_name, features_root)
    if not vdir.is_dir():
        # nested: features_root/{dataset}/{video_name or prefix-match}/
        for dataset_dir in sorted(features_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            resolved = _resolve_stem_in_dir(video_name, dataset_dir)
            candidate = dataset_dir / resolved
            if candidate.is_dir():
                vdir = candidate
                break
    if not vdir.is_dir():
        print(f"[temporal] No embedding dir found for '{video_name}' in {features_root}")
        return None
    print(f"[temporal] Loading embeddings from: {vdir}")
    paths = sorted(vdir.rglob("*.npy"))
    if not paths:
        print(f"[temporal] No .npy files in {vdir}")
        return None
    feats = []
    stem_nums = []
    _num_re = re.compile(r'(\d+)$')
    for p in paths:
        try:
            feats.append(np.load(p).astype(np.float32))
            m = _num_re.search(p.stem)
            stem_nums.append(int(m.group(1)) if m else len(stem_nums))
        except Exception as e:
            print(f"[temporal] Skip {p.name}: {e}")
    return (np.vstack(feats) if feats else None), stem_nums


def _allowed_hash(allowed_frames) -> int | None:
    """Compact fingerprint of allowed_frames list for cache invalidation."""
    if allowed_frames is None:
        return None
    return hash((len(allowed_frames), tuple(allowed_frames[:5] + allowed_frames[-5:])))


def _load_clips_cache(cache_dir: Path, clip_length: int, top_fraction: float, max_frame,
                      allowed_frames=None) -> list | None:
    """Load cached temporal clips if file exists — never invalidated unless deleted."""
    p = cache_dir / "temporal_clips.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
        clips = d.get("clips", [])
        print(f"[temporal] Loaded {len(clips)} clips from cache (immutable)")
        return clips
    except Exception as e:
        print(f"[temporal] Cache load error: {e}")
    return None


def _save_clips_cache(cache_dir: Path, clips: list, clip_length: int, top_fraction: float,
                      max_frame, allowed_frames=None):
    """Persist temporal clips to disk so they survive restarts."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "temporal_clips.json", "w") as f:
            json.dump({"clip_length": clip_length, "top_fraction": top_fraction,
                       "max_frame": max_frame, "allowed_hash": _allowed_hash(allowed_frames),
                       "clips": clips}, f)
        print(f"[temporal] Saved {len(clips)} clips to cache → {cache_dir.name}")
    except Exception as e:
        print(f"[temporal] Cache save error: {e}")


_CACHE_MISS = object()  # sentinel: value not yet cached


def _load_features_root_cache(autosave_dir: Path):
    """Return cached features root path, None (not found), or _CACHE_MISS."""
    p = autosave_dir / "features_root.json"
    if not p.exists():
        return _CACHE_MISS
    try:
        with open(p) as f:
            d = json.load(f)
        raw = d.get("path", _CACHE_MISS)
        if raw is _CACHE_MISS:
            return _CACHE_MISS
        return Path(raw) if raw is not None else None
    except Exception:
        return _CACHE_MISS


def _save_features_root_cache(autosave_dir: Path, root):
    try:
        autosave_dir.mkdir(parents=True, exist_ok=True)
        with open(autosave_dir / "features_root.json", "w") as f:
            json.dump({"path": str(root) if root is not None else None}, f)
    except Exception as e:
        print(f"[cache] features_root save error: {e}")


def _count_filtered_images(filtered_root: Path | None, video_name: str) -> int | None:
    """Count images in the filtered frames directory for cache validation."""
    if filtered_root is None:
        return None
    from pathlib import Path as _P
    # Quick resolve: try direct, then rglob (mirrors _load_allowed_frames)
    for d in [filtered_root]:
        matches = [
            m for m in d.rglob("*")
            if m.is_dir() and (video_name in m.name or (
                _trailing_digits(video_name) and _trailing_digits(video_name) in m.name))
        ]
        for folder in matches:
            n = sum(1 for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS)
            if n > 0:
                return n
    return None


def _load_allowed_frames_cache(autosave_dir: Path, expected_count: int | None = None):
    """Return cached allowed-frames list, None (no filter), or _CACHE_MISS.
    Invalidates cache if the on-disk image count has changed."""
    p = autosave_dir / "allowed_frames.json"
    if not p.exists():
        return _CACHE_MISS
    try:
        with open(p) as f:
            d = json.load(f)
        if "frames" not in d:
            return _CACHE_MISS
        if expected_count is not None and d.get("dir_count") != expected_count:
            print(f"[cache] allowed_frames stale: cached {d.get('dir_count')} vs disk {expected_count}")
            return _CACHE_MISS
        return d["frames"]  # list[int] or null → None
    except Exception:
        return _CACHE_MISS


def _save_allowed_frames_cache(autosave_dir: Path, frames, dir_count: int | None = None):
    try:
        autosave_dir.mkdir(parents=True, exist_ok=True)
        with open(autosave_dir / "allowed_frames.json", "w") as f:
            json.dump({"frames": frames, "dir_count": dir_count}, f)
    except Exception as e:
        print(f"[cache] allowed_frames save error: {e}")


def compute_temporal_clips(
        features_root: Path,
        video_name: str,
        clip_length: int = 2000,
        top_fraction: float = 0.20,
        max_frame: int | None = None,
        cache_dir: Path | None = None,
        allowed_frames: list | None = None,
) -> list:
    """
    Divide the embedding sequence into non-overlapping clips, compute a mean
    embedding per clip, then use greedy max-min diversity (L2) selection to
    pick the most spread-out top_fraction of clips.

    If allowed_frames is provided, only embeddings whose stem number is in
    that set are used — clips are built exclusively from "good" filtered
    frames.

    Selection rule:
      1. Start with the clip whose mean is furthest from the global mean.
      2. Each next pick is the clip furthest from all already-selected clips
         (maximises minimum pairwise distance).

    Short-tail correction: if the last selected clip (by frame order) has
    fewer than clip_length/2 frames, add one more diverse clip.

    Each returned dict: {"start": int, "end": int, "score": float}
    where score = L2 distance from global mean (for debug display).
    """
    if cache_dir is not None:
        cached = _load_clips_cache(cache_dir, clip_length, top_fraction, max_frame, allowed_frames)
        if cached is not None:
            return cached

    result = _load_video_embeddings(features_root, video_name)
    if result is None:
        return []
    features, stem_nums = result
    if features is None:
        return []
    n = len(features)
    print(f"[temporal] Loaded {n} embeddings, stems {stem_nums[0]}-{stem_nums[-1]}, allowed={len(allowed_frames) if allowed_frames else None}, caller={threading.current_thread().name}")
    if max_frame is not None:
        n = min(n, max_frame)
        features = features[:n]
        stem_nums = stem_nums[:n]
        print(f"[temporal] Capped at frame {max_frame} ({n} frames used)")

    # Filter to allowed frames only (skip "bad" frames removed by cleanup)
    if allowed_frames is not None:
        allowed_set = set(allowed_frames)
        keep = [i for i in range(n) if stem_nums[i] in allowed_set]
        if not keep:
            print(f"[temporal] No embeddings match allowed_frames — skipping clip computation")
            return []
        features = features[keep]
        stem_nums = [stem_nums[i] for i in keep]
        n = len(features)
        print(f"[temporal] Filtered to {n} allowed frames (from {len(allowed_set)} allowed)")

    # Build per-clip mean embeddings; start/end are stem frame numbers
    clips, means = [], []
    for arr_start in range(0, n, clip_length):
        arr_end = min(arr_start + clip_length - 1, n - 1)
        chunk = features[arr_start:arr_end + 1].astype(np.float32)
        clips.append({"start": stem_nums[arr_start], "end": stem_nums[arr_end]})
        means.append(chunk.mean(axis=0))
    if not clips:
        return []

    means = np.array(means)           # (n_clips, embed_dim)
    n_clips = len(clips)
    global_mean = means.mean(axis=0)
    dists_to_global = np.linalg.norm(means - global_mean, axis=1)

    # Determine how many clips to select
    total_frames = sum(c["end"] - c["start"] + 1 for c in clips)
    n_select = max(1, int(n_clips * top_fraction))
    # Bump if frame coverage is below target
    covered = sum(clips[i]["end"] - clips[i]["start"] + 1 for i in range(n_select))
    if covered < total_frames * top_fraction and n_select < n_clips:
        n_select += 1
        print(f"[temporal] Added 1 extra clip to reach ≥{top_fraction*100:.0f}% frame coverage")

    # Greedy max-min diversity selection
    selected = [int(np.argmax(dists_to_global))]
    while len(selected) < n_select:
        min_dists = [
            -1 if i in selected
            else min(np.linalg.norm(means[i] - means[s]) for s in selected)
            for i in range(n_clips)
        ]
        selected.append(int(np.argmax(min_dists)))

    # Short-tail correction
    top = sorted([{**clips[i], "score": float(dists_to_global[i])} for i in selected],
                 key=lambda c: c["start"])
    last_frames = top[-1]["end"] - top[-1]["start"] + 1
    if last_frames < clip_length / 2 and len(selected) < n_clips:
        remaining = [i for i in range(n_clips) if i not in selected]
        next_idx = max(remaining,
                       key=lambda i: min(np.linalg.norm(means[i] - means[s]) for s in selected))
        top.append({**clips[next_idx], "score": float(dists_to_global[next_idx])})
        top.sort(key=lambda c: c["start"])
        selected.append(next_idx)
        print(f"[temporal] Added 1 extra clip (short tail: {last_frames} < {clip_length//2} frames)")

    covered_final = sum(c["end"] - c["start"] + 1 for c in top)
    print(f"[temporal] {len(top)}/{n_clips} clips selected "
          f"(diversity sampling, coverage={covered_final}/{total_frames} "
          f"frames = {100*covered_final/total_frames:.1f}%)")
    print(f"[temporal] All clips (L2 dist from global mean):")
    selected_starts = {c["start"] for c in top}
    for i, c in enumerate(clips):
        marker = " ◀ selected" if c["start"] in selected_starts else ""
        print(f"  clip {i:>2}  frames {c['start']:>5}-{c['end']:>5}  "
              f"dist={dists_to_global[i]:.4f}{marker}")

    if cache_dir is not None:
        _save_clips_cache(cache_dir, top, clip_length, top_fraction, max_frame, allowed_frames)
    return top


# ═══════════════════════════════════════════════════════════════
#  SAM2 TRACKER  (with all extended features)
# ═══════════════════════════════════════════════════════════════
def _cap_duration_to_clip(video, start_sample: int, duration: int, clips: list) -> int:
    """Cap tracking duration so it doesn't enter a non-selected region.

    Clip start/end values are sequential 1-fps sample indices (stem numbers
    from embedding filenames like frame_000042.npy → 42), equal to the video
    sample index regardless of the video's original fps or step size.
    All comparisons therefore use start_sample directly.

    Extends coverage through consecutive selected clips (no gap between them).
    Returns duration unchanged if start_sample is not inside any clip."""
    if not clips or start_sample >= video.total:
        return duration

    # Find the clip containing start_sample
    coverage_end = None
    for clip in clips:
        if clip["start"] <= start_sample <= clip["end"]:
            coverage_end = clip["end"]
            break
    if coverage_end is None:
        return duration  # not inside any selected clip — no restriction

    # Walk forward: if the sample right after coverage_end falls inside
    # another selected clip, absorb that clip and keep going.
    changed = True
    while changed:
        changed = False
        next_sample = coverage_end + 1
        if next_sample >= video.total:
            break
        for clip in clips:
            if clip["start"] <= next_sample <= clip["end"]:
                coverage_end = clip["end"]
                changed = True
                break

    last_in_coverage = min(coverage_end, video.total - 1)
    max_frames = max(1, last_in_coverage - start_sample + 1)
    capped = min(duration, max_frames)
    if capped < duration:
        print(f"[clips] capped duration {duration}→{capped} (coverage ends at sample {coverage_end})")
    return capped


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
        self.oof       = {}     # {oid: [[start, end], ...]} — OOF ranges (end=None means open-ended)
        self.review_marks = set()  # set of sample indices marked for review
        self.blurry_ranges = []   # [[start, end], ...] — end=None means open
        self.current_frame = 0
        self.start = self.end = None
        self.running = False
        self.phase = ""
        self.progress = self.total = 0
        self.error = None


class SAM2Tracker:
    def __init__(self, autosave_dir: Path | None = None):
        self.autosave_dir = autosave_dir if autosave_dir is not None else AUTOSAVE_DIR
        self.predictor = _load_predictor()
        self.state = TrackerState()
        self._tmp = None
        self._save_timer = None
        self._save_lock = threading.Lock()
        # Parallel post-processing for multi-object tracking frames.
        self._obj_workers = max(1, min(8, os.cpu_count() or 1))
        self._obj_pool = ThreadPoolExecutor(max_workers=self._obj_workers, thread_name_prefix="sam2-obj")

    @property
    def available(self): return self.predictor is not None

    # ─── actions ──────────────────────────────────
    def start_tracking(self, video, sample_idx, boxes, duration_sec, clips=None):
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
            args=(video, sample_idx, boxes, duration_sec, next_id, clips), daemon=True)
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
            self._autosave_debounced()
            return True
        return False

    def toggle_oof(self, si, oid):
        """Toggle OOF at frame si.
        If currently OOF here (inside an open range): close that range at si.
        If not OOF here: start a new open range [si, None]."""
        s = self.state
        ranges = s.oof.get(oid, [])
        # Check if si is inside an open-ended range
        for r in ranges:
            if r[1] is None and si >= r[0]:
                # Close this range: object becomes visible again from si onwards
                r[1] = si
                self._autosave_debounced()
                return False  # no longer OOF at this frame
        # Not currently OOF — start new open range
        ranges.append([si, None])
        s.oof[oid] = ranges
        self._autosave_debounced()
        return True

    def update_nblobs(self, oid, n_blobs, from_frame=None):
        s = self.state
        if oid not in s.objects: return False
        s.objects[oid]["n_blobs"] = n_blobs
        # Recompute masks (keep only n blobs) and bboxes from current frame onwards
        for si, om in s.masks.items():
            if from_frame is not None and si < from_frame:
                continue
            if oid in om:
                filtered = mask_keep_n_blobs(om[oid], n_blobs)
                s.masks[si][oid] = filtered
                if si not in s.bboxes: s.bboxes[si] = {}
                s.bboxes[si][oid] = mask_to_bbox(filtered, n_blobs)
        self._autosave_debounced()
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
        self._autosave_debounced()
        return True

    def trim_object_from(self, oid, from_frame):
        """Remove masks/bboxes for frames >= from_frame for a single object. Keeps past data."""
        s = self.state
        if oid not in s.objects: return False
        for si in list(s.masks.keys()):
            if si >= from_frame:
                s.masks[si].pop(oid, None)
                s.bboxes.get(si, {}).pop(oid, None)
                if not s.masks[si]: del s.masks[si]
                if si in s.bboxes and not s.bboxes[si]: del s.bboxes[si]
        # Update obj_end to last remaining frame with data
        remaining = [si for si in s.masks if oid in s.masks[si]]
        s.objects[oid]["obj_end"] = max(remaining) if remaining else s.objects[oid].get("obj_start")
        # Trim OOF ranges: remove ranges starting at/after from_frame, clip ranges overlapping
        if oid in s.oof:
            new_ranges = []
            for r in s.oof[oid]:
                if r[0] >= from_frame:
                    continue  # drop entirely
                if r[1] is None or r[1] > from_frame:
                    new_ranges.append([r[0], from_frame])
                else:
                    new_ranges.append(r)
            if new_ranges:
                s.oof[oid] = new_ranges
            else:
                del s.oof[oid]
        self._recalc_range()
        self._autosave_debounced()
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
    def extend_object(self, video, oid, duration_sec, clips=None):
        if self.state.running: return
        s = self.state
        s.running = True; s.error = None
        t = threading.Thread(target=self._run_extend,
            args=(video, oid, duration_sec, clips), daemon=True)
        t.start()

    def redraw_object(self, video, oid, sample_idx, bbox, duration_sec, clips=None):
        """Trim object from sample_idx, set new bbox, and re-track forward."""
        if self.state.running: return
        s = self.state
        if oid not in s.objects: return
        s.running = True; s.error = None
        t = threading.Thread(target=self._run_redraw,
            args=(video, oid, sample_idx, bbox, duration_sec, clips), daemon=True)
        t.start()

    def _run_redraw(self, video, oid, sample_idx, bbox, duration_sec, clips):
        s = self.state
        try:
            # Trim all data from this frame onward
            for si in list(s.masks.keys()):
                if si >= sample_idx:
                    s.masks[si].pop(oid, None)
                    s.bboxes.get(si, {}).pop(oid, None)
                    if not s.masks[si]: del s.masks[si]
                    if si in s.bboxes and not s.bboxes[si]: del s.bboxes[si]
            # Trim OOF ranges from this frame
            if oid in s.oof:
                new_ranges = []
                for r in s.oof[oid]:
                    if r[0] >= sample_idx:
                        continue
                    if r[1] is None or r[1] > sample_idx:
                        new_ranges.append([r[0], sample_idx])
                    else:
                        new_ranges.append(r)
                if new_ranges:
                    s.oof[oid] = new_ranges
                else:
                    del s.oof[oid]
            # Set the new bbox at sample_idx and update obj_end so _run_extend_single works
            if sample_idx not in s.bboxes:
                s.bboxes[sample_idx] = {}
            s.bboxes[sample_idx][oid] = [int(v) for v in bbox]
            s.objects[oid]["obj_end"] = sample_idx
            # Re-track forward using extend logic
            self._run_extend_single(video, oid, duration_sec, clips)
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            s.running = False
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None
        self._autosave_background()

    def extend_all(self, video, duration_sec, oids, clips=None):
        """Extend the given objects sequentially."""
        if self.state.running: return
        s = self.state
        oids = [oid for oid in oids if oid in s.objects]
        if not oids:
            return
        s.running = True; s.error = None
        t = threading.Thread(target=self._run_extend_all,
            args=(video, oids, duration_sec, clips), daemon=True)
        t.start()

    def _run_extend_all(self, video, oids, duration_sec, clips):
        s = self.state
        total_oids = len(oids)
        try:
            for i, oid in enumerate(oids):
                s.phase = f"obj {oid} ({i+1}/{total_oids})"
                try:
                    self._run_extend_single(video, oid, duration_sec, clips)
                except Exception as e:
                    print(f"[sam2] extend_all: obj {oid} failed: {e}")
                    traceback.print_exc()
        finally:
            s.running = False
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None
        self._autosave_background()

    def _run_extend(self, video, oid, duration_sec, clips=None):
        s = self.state
        try:
            self._run_extend_single(video, oid, duration_sec, clips)
            s.running = False
            self._autosave_background()
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            s.running = False
            if self._tmp and os.path.isdir(self._tmp):
                shutil.rmtree(self._tmp, ignore_errors=True); self._tmp = None

    def _run_extend_single(self, video, oid, duration_sec, clips=None):
        """Core extend logic for a single object. Caller manages running/autosave."""
        s = self.state
        obj_info = s.objects.get(oid)
        if not obj_info:
            raise ValueError(f"Object {oid} not found")
        start_from = obj_info.get("obj_end") or s.end
        if start_from is None:
            raise ValueError(f"No end frame for object {oid}")

        bbox = s.bboxes.get(start_from, {}).get(oid)
        if bbox is None:
            for si in range(start_from, -1, -1):
                bbox = s.bboxes.get(si, {}).get(oid)
                if bbox is not None:
                    start_from = si; break
        if bbox is None:
            raise ValueError(f"No bbox found for object {oid}")

        duration_sec = _cap_duration_to_clip(video, start_from, duration_sec, clips or [])
        n_frames = min(duration_sec, video.total - start_from)
        s.total = n_frames
        new_end = start_from + n_frames - 1
        s.end = max(s.end, new_end) if s.end is not None else new_end

        nb = obj_info.get("n_blobs", 2)

        s.phase = "embedding"
        self._tmp = None
        dtype = _autocast_dtype()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            inf = self._build_inference_state(video, start_from, n_frames)
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

        obj_info["obj_end"] = new_end
        n_total = sum(1 for si in s.bboxes if oid in s.bboxes[si] and s.bboxes[si][oid] is not None)
        print(f"[sam2] Extended obj {oid} → {n_total} total bboxes (end={new_end}) ✓")

    def _is_oof(self, si, oid):
        """Check if object is OOF at given frame (inside any OOF range)."""
        ranges = self.state.oof.get(oid)
        if not ranges:
            return False
        for start, end in ranges:
            if si >= start and (end is None or si < end):
                return True
        return False

    @staticmethod
    def _mask_bbox_task(oid, raw_mask, n_blobs):
        mask = np.squeeze(raw_mask).astype(bool, copy=False)
        return oid, mask, mask_to_bbox(mask, n_blobs)

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
            # Find oof_from for the active range containing this frame
            oof_from = None
            if is_oof:
                for rng in s.oof.get(oid, []):
                    if si >= rng[0] and (rng[1] is None or si < rng[1]):
                        oof_from = rng[0]
                        break
            result[str(oid)] = {
                "label": info["label"], "bbox": box, "bbox_raw": box_raw,
                "color": rgb, "oof": is_oof,
                "oof_from": oof_from,
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
        print(f"[export] Start video export -> {path}")
        print(f"[export] State: objects={len(s.objects)} mask_frames={len(s.masks)} bbox_frames={len(s.bboxes)} start={s.start} end={s.end}")
        if not s.bboxes:
            print("[export] Aborting: no bbox data in memory (nothing to render).")
            return False
        if s.start is None or s.end is None:
            keys = sorted(s.bboxes.keys())
            if not keys:
                print("[export] Aborting: empty bbox map after validation.")
                return False
            s.start, s.end = keys[0], keys[-1]
            print(f"[export] start/end missing -> inferred range [{s.start}, {s.end}]")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
               "-s",f"{video.width}x{video.height}","-pix_fmt","bgr24",
               "-r","1","-i","-","-loglevel","error","-c:v","libx264","-pix_fmt","yuv420p",
               "-preset","fast",str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        frame_total = max(0, s.end - s.start + 1)
        frame_seen = 0
        frame_written = 0
        frame_missing = 0
        broken_pipe = False
        for si in range(s.start, s.end + 1):
            frame_seen += 1
            frame = video.get_frame(si)
            if frame is None:
                frame_missing += 1
                continue
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
                draw_label = f"#{oid} {label}"
                r, g, b = INST_COLOR.get(label, (255,255,255))
                x1,y1,x2,y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1,y1),(x2,y2),(b,g,r),3)
                (tw,th),_ = cv2.getTextSize(draw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(y1-8, th+4)
                cv2.rectangle(frame,(x1,ly-th-4),(x1+tw+8,ly+4),(b,g,r),-1)
                cv2.putText(frame, draw_label, (x1+4,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            try:
                proc.stdin.write(frame.tobytes())
                frame_written += 1
            except BrokenPipeError:
                broken_pipe = True
                break
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        print(f"[export] Frames: target={frame_total} iterated={frame_seen} written={frame_written} missing={frame_missing} broken_pipe={broken_pipe}")
        if proc.returncode != 0:
            err_text = ""
            try:
                if proc.stderr:
                    err_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
            print(f"[export] ffmpeg cmd: {' '.join(cmd)}")
            print(f"[export] ffmpeg failed ({proc.returncode}): {err_text or 'no stderr output'}")
            return False
        print("[export] ffmpeg completed successfully.")
        return True

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
    def _autosave_debounced(self):
        if self._save_timer: self._save_timer.cancel()
        self._save_timer = threading.Timer(1.0, self._autosave_background)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _autosave_background(self):
        threading.Thread(target=self._autosave, daemon=True).start()

    def _autosave(self):
        with self._save_lock:
            try:
                s = self.state
                self.autosave_dir.mkdir(parents=True, exist_ok=True)
                bboxes_ser = {}
                for si, bb in s.bboxes.items():
                    bboxes_ser[str(si)] = {str(oid): box for oid, box in bb.items()}
                oof_ser = {str(oid): ranges for oid, ranges in s.oof.items()}
                data = {
                    "objects": {str(k): v for k, v in s.objects.items()},
                    "bboxes": bboxes_ser,
                    "oof": oof_ser,
                    "review_marks": sorted(s.review_marks),
                    "blurry_ranges": s.blurry_ranges,
                    "start": s.start, "end": s.end,
                    "current_frame": s.current_frame,
                }
                with open(self.autosave_dir / "state.json", "w") as f:
                    json.dump(data, f)
                arrays = {}
                for si, om in s.masks.items():
                    for oid, mask in om.items():
                        arrays[f"s{si:06d}_o{oid}"] = mask
                if arrays:
                    np.savez_compressed(self.autosave_dir / "masks.npz", **arrays)
                print(f"[autosave] Saved {len(s.objects)} objects, {len(s.bboxes)} frames")
            except Exception as e:
                print(f"[autosave] Error: {e}")

    def _autoload(self):
        sp = self.autosave_dir / "state.json"
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
            # Load OOF - handle old formats and new ranges format
            oof_data = data.get("oof", {})
            if isinstance(oof_data, list):
                # Very old format: [[frame, oid], ...] → convert to ranges
                oof_by_oid = {}
                for si, oid in oof_data:
                    if oid not in oof_by_oid or si < oof_by_oid[oid]:
                        oof_by_oid[oid] = si
                s.oof = {oid: [[fr, None]] for oid, fr in oof_by_oid.items()}
            else:
                s.oof = {}
                for oid_str, val in oof_data.items():
                    oid = int(oid_str)
                    if isinstance(val, list) and val and isinstance(val[0], list):
                        # New ranges format: [[start, end], ...]
                        s.oof[oid] = val
                    elif isinstance(val, (int, float)):
                        # Old single-value format: frame_from → convert to open range
                        s.oof[oid] = [[int(val), None]]
                    # else: skip unknown format
            s.review_marks = set(data.get("review_marks", []))
            s.blurry_ranges = data.get("blurry_ranges", [])
            mask_path = self.autosave_dir / "masks.npz"
            if mask_path.exists():
                npz = np.load(mask_path)
                for key in npz.files:
                    parts = key.split("_o")
                    si = int(parts[0][1:])
                    oid = int(parts[1])
                    if si not in s.masks: s.masks[si] = {}
                    s.masks[si][oid] = npz[key]
            print(f"[autoload] Restored {len(s.objects)} objects, {len(s.bboxes)} frames"
                  f"{f', {len(s.review_marks)} review marks' if s.review_marks else ''}")
            return True
        except Exception as e:
            print(f"[autoload] Error: {e}")
            return False

    def _delete_autosave(self):
        if self.autosave_dir.exists():
            shutil.rmtree(self.autosave_dir, ignore_errors=True)

    def reset_for_video(self, autosave_dir: Path) -> bool:
        """Switch to a different video without reloading the SAM2 predictor."""
        if self.state.running:
            return False
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None
        self.autosave_dir = autosave_dir
        self.state = TrackerState()
        self._tmp = None
        return self._autoload()

    # ─── in-memory frame loading ──────────────────
    def _build_inference_state(self, video, start_sample, n_frames):
        """Build SAM2 inference state directly from decoded frames, skipping JPEG I/O."""
        device     = self.predictor.device
        image_size = self.predictor.image_size
        img_mean   = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        img_std    = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

        images = []
        for i in range(n_frames):
            bgr = video.get_frame(start_sample + i)
            if bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            images.append((t.to(device) - img_mean) / img_std)

        disp_w, disp_h = DISPLAY_RESOLUTION if DISPLAY_RESOLUTION else (video.width, video.height)
        inf = {
            "images":                    images,
            "num_frames":                len(images),
            "video_height":              disp_h,
            "video_width":               disp_w,
            "device":                    device,
            "storage_device":            device,
            "offload_video_to_cpu":      False,
            "offload_state_to_cpu":      False,
            "point_inputs_per_obj":      {},
            "mask_inputs_per_obj":       {},
            "cached_features":           {},
            "constants":                 {},
            "obj_id_to_idx":             OrderedDict(),
            "obj_idx_to_id":             OrderedDict(),
            "obj_ids":                   [],
            "output_dict_per_obj":       {},
            "temp_output_dict_per_obj":  {},
            "frames_tracked_per_obj":    {},
        }
        self.predictor._get_image_feature(inf, frame_idx=0, batch_size=1)
        return inf

    # ─── SAM2 tracking run ────────────────────────
    def _run(self, video, sample_idx, boxes, duration_sec, start_oid, clips=None):
        s = self.state
        try:
            duration_sec = _cap_duration_to_clip(video, sample_idx, duration_sec, clips or [])
            n_frames = min(duration_sec, video.total - sample_idx)
            s.total = n_frames
            new_start, new_end = sample_idx, sample_idx + n_frames - 1
            s.start = min(s.start, new_start) if s.start is not None else new_start
            s.end   = max(s.end, new_end) if s.end is not None else new_end

            # Build n_blobs map for bbox extraction
            n_blobs_map = {}
            for i, binfo in enumerate(boxes):
                n_blobs_map[start_oid + i] = int(binfo.get("n_blobs", 1))

            # Phase 1+2 — load frames into memory + init SAM2 state (no JPEG I/O)
            s.phase = "embedding"
            self._tmp = None
            dtype = _autocast_dtype()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                inf = self._build_inference_state(video, sample_idx, n_frames)

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
                    if hasattr(obj_ids, "cpu"):
                        obj_ids_arr = obj_ids.cpu().numpy().reshape(-1)
                    else:
                        obj_ids_arr = np.asarray(obj_ids).reshape(-1)
                    obj_ids_list = [int(v) for v in obj_ids_arr]
                    masks_np = (masks_t > 0.0).cpu().numpy()
                    n_items = min(len(obj_ids_list), len(masks_np))
                    if n_items > 1 and self._obj_workers > 1:
                        futures = []
                        for i in range(n_items):
                            oid = obj_ids_list[i]
                            nb = n_blobs_map.get(oid, s.objects.get(oid, {}).get("n_blobs", 1))
                            futures.append(self._obj_pool.submit(self._mask_bbox_task, oid, masks_np[i], nb))
                        for fut in futures:
                            oid, m, bb = fut.result()
                            s.masks[abs_si][oid] = m
                            s.bboxes[abs_si][oid] = bb
                    else:
                        for i in range(n_items):
                            oid = obj_ids_list[i]
                            nb = n_blobs_map.get(oid, s.objects.get(oid, {}).get("n_blobs", 1))
                            _, m, bb = self._mask_bbox_task(oid, masks_np[i], nb)
                            s.masks[abs_si][oid] = m
                            s.bboxes[abs_si][oid] = bb
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
            # Make results visible to frontend immediately; persist in background.
            s.running = False
            self._autosave_background()

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
.tobj .ext-btn{padding:2px 4px;border:1px solid #0a5c47;border-radius:3px;cursor:pointer;background:#073b2e;color:var(--green);transition:all .15s;display:inline-flex;align-items:center;vertical-align:middle}
.tobj .ext-btn svg{width:14px;height:14px;fill:currentColor}
.tobj .ext-btn:hover{background:#0a5240}
.tobj .ext-btn.trk-ext{color:#e74c3c;border-color:#8b2020;background:#3b1010}
.tobj .ext-btn.trk-ext:hover{background:#4d1515}
.tobj .ext-btn.disabled{background:#1a2535;border-color:var(--border);color:var(--dim);opacity:.55;cursor:not-allowed;pointer-events:none}
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
.tl-review{position:absolute;top:0;bottom:0;width:3px;background:#ff3333;z-index:5;pointer-events:none;border-radius:1px;box-shadow:0 0 4px #ff3333,0 0 8px rgba(255,51,51,.5)}
.tl-blurry{position:absolute;top:0;bottom:0;background:rgba(255,170,51,.3);border:1px solid rgba(255,170,51,.5);z-index:3;pointer-events:auto;border-radius:1px;cursor:pointer}
.blurry-preview{position:absolute;bottom:36px;transform:translateX(-50%);background:var(--bg);border:2px solid #ffaa33;border-radius:6px;padding:3px;z-index:200;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.6)}
.blurry-preview img{display:block;width:200px;border-radius:3px}
.blurry-preview .bp-label{text-align:center;font-size:10px;color:#ffaa33;padding:2px 0 0;font-family:'JetBrains Mono',monospace}
.review-active{background:#ff3333!important;border-color:#ff3333!important;color:#fff!important;animation:reviewPulse 1.2s ease-in-out infinite}
@keyframes reviewPulse{0%,100%{box-shadow:0 0 4px #ff3333}50%{box-shadow:0 0 12px #ff3333,0 0 20px rgba(255,51,51,.4)}}
.tl-lanes{position:absolute;inset:0}
.tl-lane{position:absolute;left:0;right:0}
.tl-seg{position:absolute;top:2px;bottom:2px;border-radius:4px;opacity:.45}
.tl-seg:hover{opacity:.70}
.tl-cur{position:absolute;top:0;width:2px;height:100%;background:var(--red);pointer-events:none}
.tl-temporal{position:absolute;top:0;height:100%;background:var(--accent);opacity:.13;pointer-events:none;border-radius:3px}
.pill-temp{background:#1a2636;color:var(--accent);border:1px solid #0a3a4a}
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
    <button id="reviewBtn" onclick="toggleReviewMark()" style="background:#2a1a1a;border-color:#5c2020;color:#ff6666;font-weight:700">⚑ Mark review</button>
    <button id="blurryBtn" onclick="toggleBlurry()" style="background:#2a2a1a;border-color:#5c5c20;color:#ffaa33;font-weight:700;font-size:11px">◉ start blurry</button>
    <button id="delBlurryBtn" onclick="deleteBlurry()" class="hidden" style="background:#3a1a0a;border-color:#7a3a10;color:#ff6622;font-weight:700;font-size:11px">✕ del blurry</button>
    <span class="shortcuts">A/D ±1 · W/S ±10 · Q/E ±60 · Space=play · R=review</span>
  </div>
  <div class="main-area">
    <div class="canvas-wrap"><canvas id="canvas" width="800" height="450"></canvas></div>
    <div class="side-panel">
      <div class="shdr">New Boxes (to track)</div>
      <div class="blist" id="boxList"><span class="bempty">Select instrument → draw on frame</span></div>
      <div id="trkWrap" class="hidden">
        <div class="shdr">Tracked Objects</div>
        <div class="blist" id="trkList"></div>
        <button id="trkAllBtn" class="hidden" onclick="extendAll()" style="width:100%;margin-top:4px;padding:4px 0;background:var(--accent);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;display:flex;align-items:center;justify-content:center;gap:4px"><svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:currentColor"><path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg> <span id="trkAllLabel">trk all</span></button>
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
    <label id="cbTempLbl" class="hidden" title="Restrict navigation to top temporal clips"><input type="checkbox" id="cbTemp" onchange="onTemporalToggle()"> ⚡ temporal only</label>
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
  allObjects:{},labelRects:[],bboxRanges:null,bboxRangesInFlight:false,
  temporalClips:null,temporalOnly:false,clickPoints:[],filteredFrames:null,filteredArray:null,
  reviewMarks:new Set(),blurryRanges:[]};
var canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
var frameImg=null,maskImg=null,drag=null;

fetch('/api/info').then(r=>r.json()).then(d=>{
  S.total=d.total_samples;S.vw=d.width;S.vh=d.height;S.instruments=d.instruments;
  if(d.allowed_frames&&d.allowed_frames.length){
    S.filteredFrames=new Set(d.allowed_frames);
    S.filteredArray=d.allowed_frames.slice().sort((a,b)=>a-b);
  }
  if(d.review_marks&&d.review_marks.length){
    S.reviewMarks=new Set(d.review_marks);
  }
  if(d.blurry_ranges)S.blurryRanges=d.blurry_ranges;
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
  fetchTemporalClips();
});

function selLbl(n,h){S.activeLabel=n;S.activeColor=h;
  document.querySelectorAll('.label-pill').forEach(b=>b.classList.toggle('active',b.dataset.label===n));}
function instHex(l){var i=S.instruments.find(x=>x.name===l);return i?i.hex:'#fff';}
function defaultNBlobs(label){return label==='Bipolar'?4:2;}
function rgb(a){return 'rgb('+a[0]+','+a[1]+','+a[2]+')';}

function loadImg(u){return new Promise((r,j)=>{var i=new Image();i.onload=()=>r(i);i.onerror=j;i.src=u;});}
var _loadId=0;
var _saveFrameTimer=null;
function saveFrameDebounced(){
  if(_saveFrameTimer)clearTimeout(_saveFrameTimer);
  _saveFrameTimer=setTimeout(()=>{fetch('/api/set_frame',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame})});},400);
}
function nearestAllowed(n){
  if(!S.filteredArray||!S.filteredArray.length)return n;
  var arr=S.filteredArray;
  /* binary search for insertion point */
  var lo=0,hi=arr.length-1;
  while(lo<hi){var mid=(lo+hi)>>1;if(arr[mid]<n)lo=mid+1;else hi=mid;}
  if(arr[lo]===n)return n;
  /* pick closer of arr[lo] and arr[lo-1] */
  if(lo===0)return arr[0];
  if(lo>=arr.length)return arr[arr.length-1];
  return Math.abs(arr[lo]-n)<=Math.abs(arr[lo-1]-n)?arr[lo]:arr[lo-1];
}
async function loadFrame(n){
  n=Math.max(0,Math.min(n,S.total-1));
  n=nearestAllowed(n);
  S.frame=n;
  saveFrameDebounced();
  document.getElementById('fIn').value=n;
  document.getElementById('iFrame').textContent=n;
  document.getElementById('iTime').textContent=n+'s';
  document.getElementById('tlCur').style.left=(S.total>1?(n/(S.total-1))*100:0)+'%';
  updReviewBtn();updBlurryBtn();
  var myId=++_loadId;
  var framePromise=loadImg('/api/frame/'+n+'?t='+Date.now());
  var fdPromise=Promise.resolve(S.fdCache[n]);
  if(S.tracked&&S.fdCache[n]===undefined){
    fdPromise=fetch('/api/framedata/'+n)
      .then(r=>r.ok?r.json():null)
      .catch(()=>null)
      .then(fd=>{S.fdCache[n]=fd;return fd;});
  }
  var maskPromise=Promise.resolve(null);
  if(S.tracked&&n>=S.tracked.start&&n<=S.tracked.end){
    if(S.maskCache[n]===undefined){
      maskPromise=loadImg('/api/mask/'+n)
        .then(img=>{S.maskCache[n]=img;return img;})
        .catch(()=>{S.maskCache[n]=null;return null;});
    }else{
      maskPromise=Promise.resolve(S.maskCache[n]);
    }
  }

  var newFrame;
  try{newFrame=await framePromise;}catch(e){return;}
  if(_loadId!==myId)return;/* superseded by newer loadFrame call */

  frameImg=newFrame;
  maskImg=null;
  if(canvas.width!==frameImg.naturalWidth)canvas.width=frameImg.naturalWidth;
  if(canvas.height!==frameImg.naturalHeight)canvas.height=frameImg.naturalHeight;
  redr();

  if(!S.tracked){updTrkList();updTimeline();return;}

  try{
    var res=await Promise.all([fdPromise,maskPromise]);
    if(_loadId!==myId)return;
    maskImg=res[1];
  }catch(e){
    if(_loadId!==myId)return;
    maskImg=null;
  }
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
function cCoords(e,clamp){var r=canvas.getBoundingClientRect();
  var x=(e.clientX-r.left)*(canvas.width/r.width),y=(e.clientY-r.top)*(canvas.height/r.height);
  if(clamp){x=Math.max(0,Math.min(canvas.width,x));y=Math.max(0,Math.min(canvas.height,y));}
  return{x:x,y:y};}
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
  if(drag)e.preventDefault();/* prevent native drag so mousemove keeps firing outside canvas */
});
canvas.addEventListener('mousemove',e=>{
  if(drag)return;/* handled by document listener when dragging */
  var p=cCoords(e);
  var lblHit=hitTestLabel(p.x,p.y);
  if(lblHit){canvas.style.cursor='pointer';return;}
  var h=hitTest(p.x,p.y);canvas.style.cursor=h?h.cursor:'crosshair';
});
document.addEventListener('mousemove',e=>{
  if(!drag)return;
  var p=cCoords(e,true);
  if(drag.type==='draw')drag.box=[Math.min(drag.sx,p.x),Math.min(drag.sy,p.y),Math.max(drag.sx,p.x),Math.max(drag.sy,p.y)];
  else drag.box=dragBox(p.x,p.y);
  redr();
});
function clampBox(b){return[Math.max(0,b[0]),Math.max(0,b[1]),Math.min(canvas.width,b[2]),Math.min(canvas.height,b[3])];}
document.addEventListener('mouseup',e=>{
  if(!drag)return;
  if(drag.type==='draw'){var b=drag.box?clampBox(drag.box):null;
    var isDrag=b&&(b[2]-b[0]>8)&&(b[3]-b[1]>8);
    if(isDrag){
      S.clickPoints=[];
      S.boxes.push({x1:b[0],y1:b[1],x2:b[2],y2:b[3],label:S.activeLabel,color:S.activeColor,n_blobs:defaultNBlobs(S.activeLabel)});
      updBoxList();
    } else {
      /* single click — accumulate point for 4-point bbox mode */
      S.clickPoints.push([drag.sx,drag.sy]);
      if(S.clickPoints.length===4){
        var xs=S.clickPoints.map(p=>p[0]),ys=S.clickPoints.map(p=>p[1]);
        var cb=clampBox([Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)]);
        S.clickPoints=[];
        S.boxes.push({x1:cb[0],y1:cb[1],x2:cb[2],y2:cb[3],label:S.activeLabel,color:S.activeColor,n_blobs:defaultNBlobs(S.activeLabel)});
        updBoxList();
      }
    }}
  else if(drag.box){var cb=clampBox(drag.box);S.fdCache[S.frame][drag.oid].bbox=cb;
    fetch('/api/update_bbox',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sample:S.frame,oid:parseInt(drag.oid),bbox:cb.map(v=>Math.round(v))})});}
  drag=null;updUI();redr();
});
/* drag continues outside canvas — coords clamped to border */

function isBlurry(fi){
  if(!S.blurryRanges||!S.blurryRanges.length)return false;
  return S.blurryRanges.some(r=>fi>=r[0]&&fi<=(r[1]!=null?r[1]:Infinity));
}
function redr(){
  if(!frameImg)return;ctx.drawImage(frameImg,0,0);
  S.labelRects=[];/* clear label hit areas */
  /* Blurry frame overlay */
  if(isBlurry(S.frame)){
    ctx.fillStyle='rgba(255,170,51,0.15)';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.save();ctx.font='bold 28px IBM Plex Sans,sans-serif';ctx.textAlign='right';ctx.textBaseline='top';
    ctx.fillStyle='rgba(255,170,51,0.9)';ctx.fillText('BLURRY',canvas.width-16,12);
    ctx.strokeStyle='rgba(0,0,0,0.5)';ctx.lineWidth=1;ctx.strokeText('BLURRY',canvas.width-16,12);
    ctx.restore();
  }
  if(maskImg&&document.getElementById('cbMask').checked){
    ctx.globalAlpha=0.5;ctx.drawImage(maskImg,0,0);ctx.globalAlpha=1;}
  if(document.getElementById('cbBox').checked){
    var fd=S.fdCache[S.frame];
    if(fd){for(var oid in fd){var o=fd[oid];if(!o||!o.bbox||o.oof)continue;
      var box=o.bbox;if(drag&&drag.oid===oid&&drag.box)box=drag.box;
      drawTBox(box[0],box[1],box[2],box[3],rgb(o.color),o.label,o.color,oid);}}}
  S.boxes.forEach(b=>drawNBox(b.x1,b.y1,b.x2,b.y2,b.color,b.label));
  if(drag&&drag.type==='draw'&&drag.box){var db=drag.box;drawNBox(db[0],db[1],db[2],db[3],S.activeColor,S.activeLabel);}
  if(S.clickPoints.length){
    /* draw preview bbox from accumulated points */
    var cxs=S.clickPoints.map(p=>p[0]),cys=S.clickPoints.map(p=>p[1]);
    var px1=Math.min(...cxs),py1=Math.min(...cys),px2=Math.max(...cxs),py2=Math.max(...cys);
    ctx.strokeStyle=S.activeColor||'#fff';ctx.lineWidth=2;ctx.setLineDash([6,4]);
    ctx.strokeRect(px1,py1,px2-px1,py2-py1);ctx.setLineDash([]);
    /* draw each point as numbered dot */
    S.clickPoints.forEach(function(p,i){
      ctx.beginPath();ctx.arc(p[0],p[1],6,0,2*Math.PI);
      ctx.fillStyle=S.activeColor||'#fff';ctx.fill();
      ctx.fillStyle='#000';ctx.font='bold 9px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(i+1,p[0],p[1]);ctx.textAlign='left';ctx.textBaseline='alphabetic';
    });
    /* counter label */
    ctx.font='bold 13px IBM Plex Sans,sans-serif';ctx.fillStyle=S.activeColor||'#fff';
    ctx.fillText((4-S.clickPoints.length)+' more clicks',px1+4,py1>18?py1-6:py2+16);
  }
}
function drawTBox(x1,y1,x2,y2,col,lbl,ca,oid){
  var dLbl='#'+oid+' '+lbl;
  ctx.strokeStyle=col;ctx.lineWidth=3;ctx.setLineDash([]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.fillStyle='rgba('+ca[0]+','+ca[1]+','+ca[2]+',0.08)';ctx.fillRect(x1,y1,x2-x1,y2-y1);
  ctx.font='bold 13px IBM Plex Sans,sans-serif';var tw=ctx.measureText(dLbl).width+10,th=20;
  var ly=y1>th+4?y1-th-2:y2+2;ctx.fillStyle=col;ctx.fillRect(x1,ly,tw,th);
  ctx.fillStyle='#fff';ctx.fillText(dLbl,x1+5,ly+15);
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

function fetchTemporalClips(){
  fetch('/api/temporal_clips').then(r=>r.json()).then(d=>{
    if(!d.clips||!d.clips.length){
      /* clips not ready yet (background thread still computing) — retry */
      setTimeout(fetchTemporalClips,2000);
      return;
    }
    S.temporalClips=d.clips;
    document.getElementById('cbTempLbl').classList.remove('hidden');
    /* small pill in header */
    var hdr=document.querySelector('.header');
    if(!document.getElementById('tempPill')){
      var p=document.createElement('span');
      p.id='tempPill';p.className='pill pill-temp';
      p.textContent='⚡ '+d.clips.length+' clips';
      hdr.appendChild(p);
    }
    updTimeline();
  }).catch(()=>setTimeout(fetchTemporalClips,2000));
}
function onTemporalToggle(){
  S.temporalOnly=document.getElementById('cbTemp').checked;
  if(S.temporalOnly&&S.temporalClips){
    /* snap to nearest temporal clip if currently outside all clips */
    var inClip=S.temporalClips.find(c=>S.frame>=c.start&&S.frame<=c.end);
    if(!inClip&&S.temporalClips.length)loadFrame(S.temporalClips[0].start);
  }
}

/* Returns the frame index after moving `delta` steps, respecting temporal clips when active. */
function nextTemporalFrame(from,delta){
  if(!S.temporalOnly||!S.temporalClips||!S.temporalClips.length)return from+delta;
  var clips=S.temporalClips,dir=delta>0?1:-1,steps=Math.abs(delta),cur=from;
  for(var step=0;step<steps;step++){
    var ic=clips.find(c=>cur>=c.start&&cur<=c.end);
    if(ic){
      var next=cur+dir;
      if(next>=ic.start&&next<=ic.end){cur=next;}
      else if(dir>0){
        var nc=clips.find(c=>c.start>ic.end);cur=nc?nc.start:ic.end;
      }else{
        var pc=null;for(var i=clips.length-1;i>=0;i--){if(clips[i].end<ic.start){pc=clips[i];break;}}
        cur=pc?pc.end:ic.start;
      }
    }else{
      if(dir>0){var nc2=clips.find(c=>c.start>cur);cur=nc2?nc2.start:(clips[clips.length-1]?clips[clips.length-1].end:cur);}
      else{var pc2=null;for(var i=clips.length-1;i>=0;i--){if(clips[i].end<cur){pc2=clips[i];break;}}cur=pc2?pc2.end:(clips[0]?clips[0].start:cur);}
    }
  }
  return Math.max(0,Math.min(cur,S.total-1));
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
function clearDrawn(){S.boxes=[];S.clickPoints=[];updBoxList();updUI();redr();}

function updTrkList(){
  var trkEl=document.getElementById('trkList');
  var oofEl=document.getElementById('oofList');
  var fd=S.fdCache[S.frame];
  var objs=S.allObjects;
  if(!objs||!Object.keys(objs).length){
    trkEl.innerHTML='<span class="bempty">No tracked objects</span>';
    oofEl.innerHTML='';document.getElementById('oofWrap').classList.add('hidden');return;}
  var hTrk='',hOof='',extCount=0;
  for(var oid in objs){
    var o=objs[oid],c=instHex(o.label);
    var hasMask=!!(fd&&fd[oid]&&fd[oid].bbox_raw);
    var isOof=!!(fd&&fd[oid]&&fd[oid].oof);
    var nb=(o.n_blobs!=null)?o.n_blobs:defaultNBlobs(o.label);
    var objEnd=o.obj_end!=null?o.obj_end:null;
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
      if(hasMask&&!atObjEnd&&!S.tracking){
        row+='<span class="ext-btn" onclick="redrawObj('+oid+')" title="Re-track #'+oid+' from this frame with current bbox"><svg viewBox="0 0 24 24"><path d="M17.65 6.35A7.96 7.96 0 0012 4a8 8 0 108 8h-2a6 6 0 11-1.76-4.24L14 10h7V3l-3.35 3.35z"/></svg></span>';
      }
      if(atObjEnd&&hasMask){
        extCount++;
        if(S.tracking)row+='<span class="ext-btn disabled" title="Tracking in progress"><svg viewBox="0 0 24 24"><path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg></span>';
        else row+='<span class="ext-btn trk-ext" data-oid="'+oid+'" onclick="extendObj('+oid+')" title="Extend tracking from frame '+objEnd+'"><svg viewBox="0 0 24 24"><path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg></span>';
      }
    }
    row+='<button onclick="delObj('+oid+')" title="Delete entire series">×</button></div>';
    var oofFrom=fd&&fd[oid]?fd[oid].oof_from:null;
    var showOof=isOof&&oofFrom!=null&&(S.frame-oofFrom)<=3;
    if(showOof)hOof+=row;
    else if(!isOof&&hasMask)hTrk+=row;
  }
  trkEl.innerHTML=hTrk||'<span class="bempty">All objects out of frame</span>';
  oofEl.innerHTML=hOof;
  document.getElementById('oofWrap').classList.toggle('hidden',!hOof);
  var trkAllBtn=document.getElementById('trkAllBtn');
  trkAllBtn.classList.toggle('hidden',extCount<2||S.tracking);
  document.getElementById('trkAllLabel').textContent=S.tracking?'trk all...':'trk all ('+extCount+')';
}

function updUI(){
  var hb=S.boxes.length>0,ht=S.tracked!==null;
  var trackBtn=document.getElementById('trackBtn');
  trackBtn.disabled=!hb||S.tracking;
  trackBtn.textContent=S.tracking?'tracking...':'▸ track';
  ['cbMaskLbl','cbBoxLbl','expVid','expDat','rstBtn'].forEach(id=>
    document.getElementById(id).classList.toggle('hidden',!ht));
  document.getElementById('trkWrap').classList.toggle('hidden',!ht);
  document.getElementById('progRow').classList.toggle('hidden',!S.tracking);
  if(ht)updTrkList();
}

function nextFrame(from,delta){
  /* Step through temporal clips, then skip any filtered-out frames. */
  var n=nextTemporalFrame(from,delta);
  if(!S.filteredFrames)return n;
  var dir=delta>0?1:(delta<0?-1:0);
  if(dir===0)return n;
  var limit=S.total;
  while(limit-->0&&!S.filteredFrames.has(n)){
    var n2=nextTemporalFrame(n,dir);
    if(n2===n)break; /* stuck at boundary */
    n=n2;
  }
  return n;
}
function jmp(d){loadFrame(nextFrame(S.frame,d));}
function goF(){loadFrame(+document.getElementById('fIn').value||0);}
function togPlay(){
  if(S.playTimer){clearInterval(S.playTimer);S.playTimer=null;
    document.getElementById('playBtn').textContent='▶ play';return;}
  S.playTimer=setInterval(()=>{
    var nf=nextFrame(S.frame,1);
    if(nf===S.frame||S.frame>=S.total-1){togPlay();return;}
    loadFrame(nf);
  },1000);
  document.getElementById('playBtn').textContent='⏸ stop';
}
function tlClk(e){var r=e.currentTarget.getBoundingClientRect();
  loadFrame(Math.round(((e.clientX-r.left)/r.width)*(S.total-1)));}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  var m={d:1,ArrowRight:1,a:-1,ArrowLeft:-1,w:10,ArrowUp:10,s:-10,ArrowDown:-10,q:-60,e:60};
  if(m[e.key]!==undefined){e.preventDefault();jmp(m[e.key]);}
  if(e.key===' '){e.preventDefault();togPlay();}

  if(e.key==='Escape'){S.clickPoints=[];redr();}
});

async function toggleReviewMark(){
  var r=await fetch('/api/toggle_review',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame})});
  var d=await r.json();
  if(d.ok){
    S.reviewMarks=new Set(d.review_marks);
    updTimeline();updReviewBtn();
    toast(d.marked?'⚑ Marked frame '+S.frame+' for review':'Unmarked frame '+S.frame);
  }
}
function updReviewBtn(){
  var btn=document.getElementById('reviewBtn');
  if(!btn)return;
  var marked=S.reviewMarks.has(S.frame);
  btn.textContent=marked?'⚑ MARKED':'⚑ Mark review';
  btn.classList.toggle('review-active',marked);
}
async function toggleBlurry(){
  var r=await fetch('/api/toggle_blurry',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame})});
  var d=await r.json();
  if(d.ok){
    S.blurryRanges=d.blurry_ranges;
    updTimeline();updBlurryBtn();
    toast(d.open?'◉ blurry start @ frame '+S.frame:'◉ blurry end @ frame '+S.frame);
  }
}
function updBlurryBtn(){
  var btn=document.getElementById('blurryBtn');if(!btn)return;
  var hasOpen=S.blurryRanges&&S.blurryRanges.some(r=>r[1]===null);
  btn.textContent=hasOpen?'◉ end blurry':'◉ start blurry';
  btn.style.outline=hasOpen?'2px solid #ffaa33':'none';
  var delBtn=document.getElementById('delBlurryBtn');if(!delBtn)return;
  var hasClosed=S.blurryRanges&&S.blurryRanges.some(r=>r[1]!==null);
  delBtn.classList.toggle('hidden',!hasClosed);
}
async function deleteBlurry(){
  if(!confirm('Delete all closed blurry ranges? Frames will be excluded from allowed_frames.'))return;
  var r=await fetch('/api/delete_blurry',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  var d=await r.json();
  if(d.ok){
    S.blurryRanges=d.blurry_ranges;
    if(d.allowed_frames&&d.allowed_frames.length){S.filteredFrames=new Set(d.allowed_frames);S.filteredArray=d.allowed_frames.slice().sort((a,b)=>a-b);}
    updTimeline();updBlurryBtn();
    toast('Deleted '+d.deleted_count+' blurry range(s), excluded '+d.excluded_frames+' frames');
  } else { toast(d.error||'delete failed','err'); }
}
var _bpEl=null,_bpCache={};
function showBlurryPreview(cx,topY,fi,parent){
  if(!_bpEl){
    _bpEl=document.createElement('div');_bpEl.className='blurry-preview';
    _bpEl.innerHTML='<img><div class="bp-label"></div>';
    document.body.appendChild(_bpEl);
  }
  _bpEl.style.display='block';
  _bpEl.style.left=cx+'px';_bpEl.style.top=(topY-10)+'px';
  _bpEl.style.transform='translate(-50%,-100%)';
  var img=_bpEl.querySelector('img');
  var lbl=_bpEl.querySelector('.bp-label');
  lbl.textContent='frame '+fi;
  if(_bpCache[fi]){img.src=_bpCache[fi];}
  else{var src='/api/frame/'+fi;_bpCache[fi]=src;img.src=src;}
}
function hideBlurryPreview(){if(_bpEl)_bpEl.style.display='none';}

async function startTrack(){
  if(!S.boxes.length||S.tracking)return;
  var dur=+document.getElementById('durIn').value||50;
  S.tracking=true;updUI();
  var bd=S.boxes.map(b=>({box:[b.x1,b.y1,b.x2,b.y2],label:b.label,n_blobs:(b.n_blobs!=null)?b.n_blobs:defaultNBlobs(b.label)}));
  var r=await fetch('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:S.frame,boxes:bd,duration:dur})});
  if(!r.ok){toast('Failed',true);S.tracking=false;updUI();return;}
  pollProg();
}
async function redrawObj(oid){
  if(S.tracking)return;
  var fd=S.fdCache[S.frame];
  var od=fd&&fd[oid];
  var box=od&&(od.bbox||od.bbox_raw);
  if(!box){toast('No bbox for #'+oid+' on this frame',true);return;}
  var dur=+document.getElementById('durIn').value||50;
  S.tracking=true;updUI();
  var r=await fetch('/api/redraw',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({oid:oid,frame:S.frame,box:box.map(v=>Math.round(v)),duration:dur})});
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

  /* Paint temporal clip backgrounds first (always visible, even without tracks) */
  if(S.temporalClips&&S.temporalClips.length&&S.total>0){
    S.temporalClips.forEach(c=>{
      var div=document.createElement('div');
      div.className='tl-temporal';
      div.style.left=((c.start/S.total)*100)+'%';
      div.style.width=(((c.end-c.start+1)/S.total)*100)+'%';
      div.title='Temporal clip '+c.start+'-'+c.end+' score='+c.score.toFixed(4);
      lanesEl.appendChild(div);
    });
  }

  /* Paint review marks — bright red vertical lines */
  if(S.reviewMarks&&S.reviewMarks.size&&S.total>0){
    S.reviewMarks.forEach(si=>{
      var div=document.createElement('div');
      div.className='tl-review';
      div.style.left=((si/S.total)*100)+'%';
      div.title='⚑ Review frame '+si;
      lanesEl.appendChild(div);
    });
  }
  /* Paint blurry ranges — orange bars with hover preview */
  if(S.blurryRanges&&S.blurryRanges.length&&S.total>0){
    S.blurryRanges.forEach(r=>{
      var s0=r[0],s1=r[1]!=null?r[1]:S.total;
      var div=document.createElement('div');
      div.className='tl-blurry';
      div.style.left=((s0/S.total)*100)+'%';
      div.style.width=(((s1-s0)/S.total)*100)+'%';
      div.title='';
      div.addEventListener('mousemove',function(ev){
        var rect=this.getBoundingClientRect();
        var frac=(ev.clientX-rect.left)/rect.width;
        var fi=Math.round(s0+frac*(s1-s0));
        fi=Math.max(s0,Math.min(fi,s1));
        showBlurryPreview(ev.clientX,rect.top,fi,this);
      });
      div.addEventListener('mouseleave',hideBlurryPreview);
      div.addEventListener('click',function(ev){
        ev.stopPropagation();
        var rect=this.getBoundingClientRect();
        var frac=(ev.clientX-rect.left)/rect.width;
        var fi=Math.round(s0+frac*(s1-s0));
        loadFrame(Math.max(s0,Math.min(fi,s1)));
      });
      lanesEl.appendChild(div);
    });
  }

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
    body:JSON.stringify({oid:oid,n_blobs:n,frame:S.frame})});
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
async function extendAll(){
  if(S.tracking)return;
  /* collect OIDs from the visible ▸trk buttons in the DOM */
  var oids=[];
  document.querySelectorAll('#trkList .trk-ext').forEach(function(b){oids.push(+b.dataset.oid);});
  if(!oids.length)return;
  var dur=+document.getElementById('durIn').value||50;
  S.tracking=true;updUI();
  var r=await fetch('/api/extend_all',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({duration:dur,oids:oids})});
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
#  VIDEO PICKER
# ═══════════════════════════════════════════════════════════════

_picker_cache: dict = {}  # video_stem -> computed info dict
_picker_computing: set = set()  # stems currently being computed in background


def _get_picker_video_paths() -> list[Path]:
    """Return sorted list of video files in VIDEO_PICKER_DIR."""
    if VIDEO_PICKER_DIR is None:
        return []
    vdir = Path(VIDEO_PICKER_DIR)
    if not vdir.is_dir():
        return []
    exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
    return sorted(p for p in vdir.iterdir() if p.suffix.lower() in exts)


def _compute_picker_info(video_path: Path) -> dict:
    """Compute completion stats for one video (may take a few seconds)."""
    stem = video_path.stem
    autosave_dir = EXPORT_DIR / "autosave" / _resolve_stem_in_dir(stem, EXPORT_DIR / "autosave")
    state_json = autosave_dir / "state.json"
    info = {
        "path": str(video_path), "name": stem,
        "started": False, "object_count": 0, "tracked_frames": 0,
        "total_clips": 0, "annotated_clips": 0, "completion": 0.0,
        "clips_ready": False,
    }
    bboxes = {}
    if state_json.exists():
        try:
            with open(state_json) as f:
                state = json.load(f)
            objects = state.get("objects", {})
            bboxes = state.get("bboxes", {})
            info["started"] = bool(objects)
            info["object_count"] = len(objects)
            info["tracked_frames"] = len(bboxes)
        except Exception as e:
            print(f"[picker] Error reading {stem}: {e}")
    feat_root = Path(FEATURES_ROOT) if FEATURES_ROOT is not None else _find_features_root(video_path)
    if feat_root is None:
        return info
    try:
        clips = compute_temporal_clips(
            feat_root, stem,
            clip_length=CLIP_LENGTH,
            top_fraction=TEMPORAL_TOP_FRACTION,
            max_frame=TEMPORAL_MAX_FRAME,
            cache_dir=autosave_dir,
        )
        if clips:
            bbox_frames = {int(k) for k in bboxes.keys()}
            annotated = sum(
                1 for c in clips
                if any(c["start"] <= f <= c["end"] for f in bbox_frames)
            )
            info["total_clips"] = len(clips)
            info["annotated_clips"] = annotated
            info["completion"] = annotated / len(clips)
            info["clips_ready"] = True
    except Exception as e:
        print(f"[picker] Temporal clip error for {stem}: {e}")
    return info


_LAST_VIDEO_FILE = EXPORT_DIR / "autosave" / "last_video.json"


def _load_last_video() -> Path | None:
    """Return the path of the last used video, or None if not set / missing."""
    try:
        if _LAST_VIDEO_FILE.exists():
            with open(_LAST_VIDEO_FILE) as f:
                d = json.load(f)
            p = Path(d["path"])
            return p if p.exists() else None
    except Exception:
        pass
    return None


def _save_last_video(path: Path):
    """Persist the last used video path so it can be restored on next launch."""
    try:
        _LAST_VIDEO_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LAST_VIDEO_FILE, "w") as f:
            json.dump({"path": str(path)}, f)
    except Exception as e:
        print(f"[picker] Could not save last video: {e}")


PICKER_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Picker</title>
<style>
:root{--bg:#0c1017;--panel:#151c28;--border:#1e2a3a;--border-h:#2a3a50;--text:#c8d6e5;--dim:#5a6d82;--accent:#00b4d8;--green:#06d6a0;--red:#ef476f;--orange:#ffd166}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:24px}
.header{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:20px}
.header h1{font-family:monospace;font-size:15px;font-weight:600;color:var(--accent);letter-spacing:.5px}
.header a{font-size:12px;color:var(--accent);text-decoration:none;margin-left:auto;padding:5px 12px;border:1px solid var(--border);border-radius:4px}
.header a:hover{border-color:var(--accent)}
.grid{display:flex;flex-direction:column;gap:6px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:12px 16px;cursor:pointer;transition:border-color .15s,background .15s;display:flex;align-items:center;gap:14px}
.card:hover{border-color:var(--border-h);background:#1a2535}
.card.current{border-color:var(--accent);cursor:default}
.cur-arrow{color:var(--accent);font-size:16px;flex-shrink:0}
.name{font-family:monospace;font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:10px;padding:2px 8px;border-radius:99px;font-weight:600;flex-shrink:0;letter-spacing:.3px}
.b-new{color:var(--dim);background:#1a2535;border:1px solid var(--border)}
.b-started{color:var(--orange);background:#2a2010;border:1px solid #4a3820}
.b-done{color:var(--green);background:#073b2e;border:1px solid #0a5c47}
.b-current{color:var(--accent);background:#0a2030;border:1px solid var(--accent)}
.prog-wrap{width:100px;background:#1a2535;border-radius:3px;height:5px;flex-shrink:0;overflow:hidden}
.prog-bar{height:100%;background:var(--green);transition:width .4s}
.pct{font-size:11px;color:var(--text);font-family:monospace;min-width:34px;text-align:right;flex-shrink:0}
.clips-info{font-size:11px;color:var(--dim);font-family:monospace;min-width:70px;text-align:right;flex-shrink:0}
.computing{font-size:11px;color:var(--dim);font-style:italic;flex-shrink:0}
</style></head><body>
<div class="header">
  <h1>Video Picker</h1>
  <a href="/">→ Tracker</a>
</div>
<div class="grid" id="grid">Loading…</div>
<script>
async function load() {
  var r, data;
  try { r = await fetch('/api/picker_videos'); data = await r.json(); } catch(e) { return; }
  var videos = data.videos || [];
  var g = document.getElementById('grid');
  if (!videos.length) { g.textContent = 'No videos found. Set VIDEO_PICKER_DIR in the script config.'; return; }
  var scrollY = window.scrollY;
  g.innerHTML = '';
  for (var i = 0; i < videos.length; i++) {
    var v = videos[i];
    var pct = v.clips_ready ? Math.round(v.completion * 100) : null;
    var isDone = pct !== null && pct >= 90;
    var badgeClass = v.current ? 'b-current' : (v.started ? (isDone ? 'b-done' : 'b-started') : 'b-new');
    var badgeText = v.current ? 'current' : (v.started ? (isDone ? 'done' : 'started') : 'new');
    var card = document.createElement('div');
    card.className = 'card' + (v.current ? ' current' : '');
    var html = '';
    if (v.current) html += '<span class="cur-arrow">&#9658;</span>';
    html += '<div class="name" title="' + v.name + '">' + v.name + '</div>';
    html += '<span class="badge ' + badgeClass + '">' + badgeText + '</span>';
    if (v.clips_ready) {
      html += '<div class="prog-wrap"><div class="prog-bar" style="width:' + pct + '%"></div></div>';
      html += '<div class="pct">' + pct + '%</div>';
      html += '<div class="clips-info">' + v.annotated_clips + '/' + v.total_clips + ' clips</div>';
    } else if (v.started) {
      html += '<div class="clips-info">' + v.tracked_frames + ' frames tracked</div>';
      html += '<div class="computing">computing clips…</div>';
    } else {
      html += '<div class="clips-info">—</div>';
    }
    card.innerHTML = html;
    (function(vv, cc) {
      if (!vv.current) {
        cc.onclick = async function() {
          cc.style.opacity = '0.5';
          var resp = await fetch('/api/switch_video', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: vv.path})
          });
          var res = await resp.json();
          if (res.ok) { window.location.href = '/'; }
          else { alert('Switch failed: ' + (res.error || 'unknown')); cc.style.opacity = '1'; }
        };
      } else {
        cc.onclick = function() { window.location.href = '/'; };
      }
    })(v, card);
    g.appendChild(card);
  }
  window.scrollTo(0, scrollY);
}
load();
setInterval(load, 4000);
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════
#  VERIFY: show extracted frames with mask overlay
# ═══════════════════════════════════════════════════════════════

def _find_extracted_frames_dir() -> Path | None:
    """Find the extracted frames directory for the current video."""
    if VIDEO_PATH is None or EXTRACTED_FRAMES_ROOT is None:
        return None
    stem = Path(VIDEO_PATH).stem
    resolved = _resolve_stem_in_dir(stem, EXTRACTED_FRAMES_ROOT)
    d = EXTRACTED_FRAMES_ROOT / resolved
    if d.is_dir() and any(d.iterdir()):
        return d
    # Search one level deeper (dataset/video_name)
    for sub in EXTRACTED_FRAMES_ROOT.iterdir():
        if not sub.is_dir():
            continue
        resolved = _resolve_stem_in_dir(stem, sub)
        d = sub / resolved
        if d.is_dir() and any(d.iterdir()):
            return d
    return None


def _get_extracted_frame_jpeg(sample_idx: int, q=85) -> bytes | None:
    """Load extracted frame from disk by sample index, return JPEG bytes."""
    d = _find_extracted_frames_dir()
    if d is None:
        return None
    # Find frame file matching this sample index
    patterns = [f"frame_{sample_idx:06d}.*", f"*{sample_idx:06d}.*"]
    for pat in patterns:
        matches = list(d.glob(pat))
        matches = [m for m in matches if m.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
        if matches:
            img = cv2.imread(str(matches[0]))
            if img is not None:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                return buf.tobytes()
    return None


def _build_verify_html() -> str:
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Verify Labels vs Extracted Frames</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column}
#bar{padding:10px 16px;background:#1a1a1a;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}
#counter{font-weight:bold;font-size:1.1em}
#info{color:#888;font-size:.9em}
#hint{font-size:.8em;color:#555;margin-left:auto}
#view{flex:1;display:flex;gap:0;min-height:0;position:relative}
#view canvas{flex:1;width:100%;height:100%;object-fit:contain}
.toggle-row{padding:6px 16px;background:#1a1a1a;display:flex;gap:12px;align-items:center;font-size:.85em}
.toggle-row label{cursor:pointer;color:#aaa}
.toggle-row input{margin-right:4px}
</style></head><body>
<div id="bar">
  <span id="counter">loading...</span>
  <span id="info"></span>
  <span id="hint">a/d or arrows: navigate | b: toggle boxes | s: toggle side-by-side</span>
</div>
<div class="toggle-row">
  <label><input type="checkbox" id="cbBox" checked> Show bounding boxes</label>
  <label><input type="checkbox" id="cbSide"> Side-by-side (extracted vs mp4)</label>
</div>
<div id="view"><canvas id="c"></canvas></div>
<script>
var frames=[],objs={},idx=0,showBox=true,showSide=false,frameData=null;
var c=document.getElementById('c'),ctx=c.getContext('2d');
var imgE=new Image(),imgV=new Image();
var loadedE=false,loadedV=false;

fetch('/api/verify_info').then(r=>r.json()).then(d=>{
  frames=d.frames;objs=d.objects;
  if(frames.length)show();
  else document.getElementById('counter').textContent='No labelled frames';
});

function show(){
  var si=frames[idx];
  document.getElementById('counter').textContent=(idx+1)+'/'+frames.length+' (sample '+si+')';
  loadedE=loadedV=false;frameData=null;
  imgE.onload=function(){loadedE=true;draw();};
  imgV.onload=function(){loadedV=true;draw();};
  imgE.onerror=function(){loadedE=true;draw();};
  imgV.onerror=function(){loadedV=true;draw();};
  imgE.src='/api/verify_frame/'+si+'?t='+Date.now();
  imgV.src='/api/frame/'+si+'?t='+Date.now();
  fetch('/api/framedata/'+si).then(r=>r.json()).then(d=>{frameData=d;draw();});
}

function imgRect(img,x,y,w,h){
  if(!img.naturalWidth)return null;
  var ar=img.naturalWidth/img.naturalHeight;
  var dw=w,dh=h;
  if(dw/dh>ar){dw=dh*ar;}else{dh=dw/ar;}
  return{dx:x+(w-dw)/2,dy:y+(h-dh)/2,dw:dw,dh:dh,
         sx:dw/img.naturalWidth,sy:dh/img.naturalHeight};
}

function draw(){
  if(!loadedE)return;
  var W=c.parentElement.clientWidth,H=c.parentElement.clientHeight;
  c.width=W;c.height=H;
  ctx.fillStyle='#111';ctx.fillRect(0,0,W,H);
  if(showSide){
    drawPanel(imgE,0,0,W/2,H,'EXTRACTED');
    if(loadedV)drawPanel(imgV,W/2,0,W/2,H,'MP4');
  } else {
    drawPanel(imgE,0,0,W,H,'EXTRACTED');
  }
}

function drawPanel(img,x,y,w,h,label){
  var r=imgRect(img,x,y,w,h);
  if(!r)return;
  ctx.drawImage(img,r.dx,r.dy,r.dw,r.dh);
  if(label){ctx.fillStyle='rgba(0,0,0,0.5)';ctx.fillRect(r.dx,r.dy,80,20);
    ctx.fillStyle='#fff';ctx.font='12px sans-serif';ctx.fillText(label,r.dx+4,r.dy+14);}
  if(showBox&&frameData){
    /* bbox coords are in mp4 resolution — scale relative to mp4 size, not panel image size */
    var refW=imgV.naturalWidth||img.naturalWidth, refH=imgV.naturalHeight||img.naturalHeight;
    var bsx=r.dw/refW, bsy=r.dh/refH;
    var labels=[];
    for(var oid in frameData){
      var od=frameData[oid];
      if(!od.bbox||od.oof)continue;
      var bb=od.bbox;
      var clr='rgb('+od.color[0]+','+od.color[1]+','+od.color[2]+')';
      var bx=r.dx+bb[0]*bsx,by=r.dy+bb[1]*bsy;
      var bw=(bb[2]-bb[0])*bsx,bh=(bb[3]-bb[1])*bsy;
      ctx.strokeStyle=clr;ctx.lineWidth=2;ctx.strokeRect(bx,by,bw,bh);
      ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(bx,by-16,ctx.measureText(od.label).width+8,16);
      ctx.fillStyle=clr;ctx.font='bold 12px sans-serif';ctx.fillText(od.label,bx+4,by-4);
      labels.push(od.label);
    }
    document.getElementById('info').textContent=labels.join(', ');
  }
}

document.addEventListener('keydown',function(e){
  if(e.key==='d'||e.key==='ArrowRight'){if(idx<frames.length-1){idx++;show();}}
  else if(e.key==='a'||e.key==='ArrowLeft'){if(idx>0){idx--;show();}}
  else if(e.key==='b'){showBox=!showBox;document.getElementById('cbBox').checked=showBox;draw();}
  else if(e.key==='s'){showSide=!showSide;document.getElementById('cbSide').checked=showSide;draw();}
});
document.getElementById('cbBox').onchange=function(){showBox=this.checked;draw();};
document.getElementById('cbSide').onchange=function(){showSide=this.checked;draw();};
window.addEventListener('resize',draw);
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════
#  HTTP SERVER
# ═══════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    video:           object       = None   # VideoReader or FrameReader
    tracker:         SAM2Tracker  = None
    temporal_clips:  list         = []
    allowed_frames:  object       = None   # list[int] or None
    _setup_gen:      int          = 0      # incremented on each video switch; stale threads abort
    _refreshing:     bool         = False  # guard against concurrent refreshes

    @classmethod
    def _maybe_refresh_frames(cls):
        """If FrameReader source dir changed (frames deleted/added), rebuild
        the file list and recompute allowed_frames.  Called on /api/info."""
        vid = cls.video
        if vid is None or not hasattr(vid, 'refresh') or cls._refreshing:
            return
        if not vid.refresh():
            return  # nothing changed
        cls._refreshing = True
        try:
            vp = VIDEO_PATH
            adir = cls.tracker.autosave_dir if cls.tracker else None
            frames = _load_allowed_frames(FILTERED_FRAMES_ROOT, vid, vp.stem if vp else None)
            frames = _apply_custom_exclusions(frames, adir)
            cls.allowed_frames = frames
            ic = vid.total  # new count IS the source dir count for FrameReader
            _save_allowed_frames_cache(adir, frames, _count_filtered_images(FILTERED_FRAMES_ROOT, vp.stem) if vp else None)
            print(f"[refresh] allowed_frames recomputed → {len(frames) if frames else 'none'}")
        except Exception as e:
            print(f"[refresh] error: {e}")
        finally:
            cls._refreshing = False

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            if Handler.video is None:
                self._send(200, "text/html", PICKER_HTML.encode())
            else:
                self._send(200, "text/html", HTML_PAGE.encode())
        elif p == "/picker":
            self._send(200, "text/html", PICKER_HTML.encode())
        elif p == "/api/picker_videos":
            videos = _get_picker_video_paths()
            current_stem = Path(VIDEO_PATH).stem if VIDEO_PATH is not None else ""
            result = []
            for vp in videos:
                stem = vp.stem
                if stem in _picker_cache:
                    info = dict(_picker_cache[stem])
                else:
                    info = {
                        "path": str(vp), "name": stem,
                        "started": (EXPORT_DIR / "autosave" / _resolve_stem_in_dir(stem, EXPORT_DIR / "autosave") / "state.json").exists(),
                        "object_count": 0, "tracked_frames": 0,
                        "total_clips": 0, "annotated_clips": 0,
                        "completion": 0.0, "clips_ready": False,
                    }
                    if stem not in _picker_computing:
                        _picker_computing.add(stem)
                        def _bg(vp=vp):
                            try:
                                res = _compute_picker_info(vp)
                                _picker_cache[vp.stem] = res
                            finally:
                                _picker_computing.discard(vp.stem)
                        threading.Thread(target=_bg, daemon=True).start()
                info["current"] = (stem == current_stem)
                result.append(info)
            self._json({"videos": result})
        elif p == "/api/info":
            self._maybe_refresh_frames()
            s = self.tracker.state
            has_tracked = s.start is not None
            self._json({
                "total_samples": self.video.total,
                "sample_fps": SAMPLE_FPS,
                "step": self.video.step,
                "orig_fps": self.video.orig_fps,
                "width": DISPLAY_RESOLUTION[0] if DISPLAY_RESOLUTION else self.video.width,
                "height": DISPLAY_RESOLUTION[1] if DISPLAY_RESOLUTION else self.video.height,
                "sam2_ok": self.tracker.available,
                "instruments": INSTRUMENTS,
                "has_tracked": has_tracked,
                "tracked_start": s.start,
                "tracked_end": s.end,
                "current_frame": s.current_frame,
                "objects": {str(k): v for k, v in s.objects.items()} if has_tracked else {},
                "allowed_frames": self.allowed_frames,
                "review_marks": sorted(self.tracker.state.review_marks),
                "blurry_ranges": self.tracker.state.blurry_ranges,
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
        elif p == "/api/temporal_clips":
            self._json({"clips": Handler.temporal_clips or []})
        elif p == "/verify":
            self._send(200, "text/html", _build_verify_html().encode())
        elif p.startswith("/api/verify_frame/"):
            si = int(p.split("/")[-1])
            img = _get_extracted_frame_jpeg(si)
            self._send(200, "image/jpeg", img) if img else self.send_error(404)
        elif p == "/api/verify_info":
            s = self.tracker.state
            labelled = sorted(set(s.masks.keys()) | set(s.bboxes.keys()))
            self._json({"frames": labelled, "objects": {str(k): v for k, v in s.objects.items()}})
        else:
            self.send_error(404)

    def do_POST(self):
        global VIDEO_PATH
        p = self.path.split("?")[0]
        body = self._body()
        if p == "/api/track":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.start_tracking(self.video, int(d["frame"]),
                d["boxes"], int(d.get("duration", TRACK_SECONDS)),
                clips=Handler.temporal_clips or None)
            self._json({"ok": True})
        elif p == "/api/extend_track":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.extend_object(self.video, int(d["oid"]),
                int(d.get("duration", TRACK_SECONDS)),
                clips=Handler.temporal_clips or None)
            self._json({"ok": True})
        elif p == "/api/extend_all":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            oids = [int(o) for o in d.get("oids", [])]
            self.tracker.extend_all(self.video,
                int(d.get("duration", TRACK_SECONDS)),
                oids=oids,
                clips=Handler.temporal_clips or None)
            self._json({"ok": True})
        elif p == "/api/redraw":
            if not self.tracker.available:
                self._json({"ok": False, "error": "SAM2 not loaded"}, 503); return
            d = json.loads(body)
            self.tracker.redraw_object(self.video, int(d["oid"]),
                int(d["frame"]), d["box"],
                int(d.get("duration", TRACK_SECONDS)),
                clips=Handler.temporal_clips or None)
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
            ok = self.tracker.update_nblobs(int(d["oid"]), int(d["n_blobs"]),
                from_frame=int(d["frame"]) if "frame" in d else None)
            self._json({"ok": ok})
        elif p == "/api/set_frame":
            d = json.loads(body)
            self.tracker.state.current_frame = int(d.get("frame", 0))
            self._json({"ok": True})
        elif p == "/api/toggle_review":
            d = json.loads(body)
            si = int(d["frame"])
            s = self.tracker.state
            if si in s.review_marks:
                s.review_marks.discard(si)
                marked = False
            else:
                s.review_marks.add(si)
                marked = True
            self.tracker._autosave_debounced()
            self._json({"ok": True, "marked": marked, "review_marks": sorted(s.review_marks)})
        elif p == "/api/toggle_blurry":
            d = json.loads(body)
            si = int(d["frame"])
            s = self.tracker.state
            # Check if there's an open range to close
            for r in s.blurry_ranges:
                if r[1] is None:
                    r[1] = si
                    self.tracker._autosave_debounced()
                    self._json({"ok": True, "open": False, "blurry_ranges": s.blurry_ranges})
                    return
            # No open range — start a new one
            s.blurry_ranges.append([si, None])
            self.tracker._autosave_debounced()
            self._json({"ok": True, "open": True, "blurry_ranges": s.blurry_ranges})
        elif p == "/api/delete_blurry":
            s = self.tracker.state
            # Collect closed blurry ranges and append to custom.txt
            closed = [r for r in s.blurry_ranges if r[1] is not None]
            if not closed:
                self._json({"ok": False, "error": "no closed blurry ranges"})
                return
            adir = self.tracker.autosave_dir
            adir.mkdir(parents=True, exist_ok=True)
            custom = adir / "custom.txt"
            with open(custom, "a") as f:
                for r in closed:
                    f.write(f"{r[0]} {r[1]}\n")
            # Remove closed ranges from state (keep open ones)
            s.blurry_ranges = [r for r in s.blurry_ranges if r[1] is None]
            self.tracker._autosave_debounced()
            # Refresh allowed_frames
            vp = VIDEO_PATH
            frames = _load_allowed_frames(FILTERED_FRAMES_ROOT, self.video, vp.stem if vp else None)
            frames = _apply_custom_exclusions(frames, adir)
            Handler.allowed_frames = frames
            _save_allowed_frames_cache(adir, frames, _count_filtered_images(FILTERED_FRAMES_ROOT, vp.stem) if vp else None)
            n_excluded = sum(r[1] - r[0] + 1 for r in closed)
            print(f"[blurry] Deleted {len(closed)} ranges ({n_excluded} frames) → custom.txt")
            self._json({
                "ok": True,
                "blurry_ranges": s.blurry_ranges,
                "allowed_frames": Handler.allowed_frames,
                "deleted_count": len(closed),
                "excluded_frames": n_excluded,
            })
        elif p == "/api/delete_object":
            d = json.loads(body)
            ok = self.tracker.delete_object(int(d["oid"]))
            self._json({"ok": ok})
        elif p == "/api/trim_object":
            d = json.loads(body)
            ok = self.tracker.trim_object_from(int(d["oid"]), int(d["from_frame"]))
            self._json({"ok": ok})
        elif p == "/api/reset":
            self.tracker.reset()
            self._json({"ok": True})
        elif p == "/api/switch_video":
            d = json.loads(body)
            vpath = Path(d["path"])
            if not vpath.exists():
                self._json({"ok": False, "error": "Video not found"}); return
            try:
                new_video = make_video_source(vpath, SAMPLE_FPS)
                new_autosave_dir = EXPORT_DIR / "autosave" / _resolve_stem_in_dir(vpath.stem, EXPORT_DIR / "autosave")
                Handler.tracker.reset_for_video(new_autosave_dir)
                Handler.video = new_video
                Handler.temporal_clips = []
                VIDEO_PATH = vpath
                _save_last_video(vpath)

                # Load caches — these are tiny JSON files, fast even on first hit
                feat_root_cached  = _load_features_root_cache(new_autosave_dir)
                _img_count = _count_filtered_images(FILTERED_FRAMES_ROOT, vpath.stem)
                allowed_cached    = _load_allowed_frames_cache(new_autosave_dir, _img_count)

                # Apply whatever we already know immediately
                Handler.allowed_frames = allowed_cached if allowed_cached is not _CACHE_MISS else None

                # Bump generation so any previous _bg_setup thread will abort
                Handler._setup_gen += 1
                my_gen = Handler._setup_gen

                # Background: resolve any cache misses, then compute temporal clips.
                # Checks generation at each step — exits immediately if a newer
                # switch happened. Heavy HDD I/O also waits for tracking to finish.
                def _bg_setup(vp=vpath, vid=new_video, adir=new_autosave_dir,
                               frc=feat_root_cached, ac=allowed_cached, gen=my_gen, ic=_img_count):
                    has_miss = frc is _CACHE_MISS or ac is _CACHE_MISS
                    if has_miss:
                        while Handler.tracker.state.running:
                            if Handler._setup_gen != gen: return
                            time.sleep(1)
                    if Handler._setup_gen != gen: return
                    feat_root = frc
                    if frc is _CACHE_MISS:
                        feat_root = Path(FEATURES_ROOT) if FEATURES_ROOT is not None \
                                    else _find_features_root(vp)
                        if Handler._setup_gen != gen: return
                        _save_features_root_cache(adir, feat_root)
                        print(f"[cache] features_root → {feat_root}")
                    if ac is _CACHE_MISS:
                        frames = _load_allowed_frames(FILTERED_FRAMES_ROOT, vid, vp.stem)
                        frames = _apply_custom_exclusions(frames, adir)
                        if Handler._setup_gen != gen: return
                        Handler.allowed_frames = frames
                        _save_allowed_frames_cache(adir, frames, ic)
                        print(f"[cache] allowed_frames → {len(frames) if frames else 'none'}")
                    if Handler._setup_gen != gen: return
                    if feat_root is not None:
                        Handler.temporal_clips = compute_temporal_clips(
                            feat_root, vp.stem,
                            clip_length=CLIP_LENGTH,
                            top_fraction=TEMPORAL_TOP_FRACTION,
                            max_frame=TEMPORAL_MAX_FRAME,
                            cache_dir=adir,
                            allowed_frames=Handler.allowed_frames,
                        )
                threading.Thread(target=_bg_setup, daemon=True).start()
                self._json({"ok": True})
            except Exception as e:
                self._json({"ok": False, "error": str(e)})
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
    # Load SAM2 predictor upfront (this is slow — done once, reused across video switches)
    _startup_autosave = EXPORT_DIR / "autosave" / "_no_video"
    tracker = SAM2Tracker(autosave_dir=_startup_autosave)
    if not tracker.available:
        print("\n  ⚠  SAM2 NOT FOUND")
        print("  Install:  pip install sam2   # or:")
        print("            git clone https://github.com/facebookresearch/sam2.git ~/sam2")
        print("            cd ~/sam2 && pip install -e .\n")

    # Restore last used video, or start with picker if none saved
    video = None
    _last_video = _load_last_video()
    if _last_video is not None:
        print(f"[main] Restoring last video: {_last_video.name}")
        try:
            video = make_video_source(_last_video, sample_fps=SAMPLE_FPS)
            autosave_dir = EXPORT_DIR / "autosave" / _resolve_stem_in_dir(_last_video.stem, EXPORT_DIR / "autosave")
            autoload_ok = tracker.reset_for_video(autosave_dir)
            VIDEO_PATH = _last_video
            if autoload_ok:
                print("[main] Restored previous tracking session")
        except Exception as e:
            print(f"[main] Could not load last video: {e}")
            video = None

    if video is None:
        print(f"[main] No video loaded — open http://localhost:{PORT} to pick one")

    if "--export-labeled" in sys.argv:
        if video is None:
            print("[main] --export-labeled requires a video to be loaded (select one via picker first)")
            sys.exit(1)
        s = tracker.state
        print(f"[main] Export debug state: objects={len(s.objects)} mask_frames={len(s.masks)} bbox_frames={len(s.bboxes)} start={s.start} end={s.end}")
        out_path = EXPORT_DIR / "ALTR-20.mp4"
        ok = tracker.export_video(video, str(out_path))
        print(f"[export] {'OK' if ok else 'FAIL'} → {out_path}")
        sys.exit(0)

    # Resolve caches and spin up background thread if a video is loaded
    Handler.temporal_clips = []
    _startup_allowed = None
    if video is not None:
        _startup_autosave_dir = tracker.autosave_dir  # capture before Handler assignment
        feat_root_cached  = _load_features_root_cache(_startup_autosave_dir)
        _img_count = _count_filtered_images(FILTERED_FRAMES_ROOT, _last_video.stem)
        allowed_cached    = _load_allowed_frames_cache(_startup_autosave_dir, _img_count)
        _startup_allowed  = allowed_cached if allowed_cached is not _CACHE_MISS else None

        def _bg_startup(vp=_last_video, vid=video, adir=_startup_autosave_dir,
                        frc=feat_root_cached, ac=allowed_cached, ic=_img_count):
            has_miss = frc is _CACHE_MISS or ac is _CACHE_MISS
            if has_miss:
                while Handler.tracker is not None and Handler.tracker.state.running:
                    time.sleep(1)
            feat_root = frc
            if frc is _CACHE_MISS:
                feat_root = Path(FEATURES_ROOT) if FEATURES_ROOT is not None \
                            else _find_features_root(vp)
                _save_features_root_cache(adir, feat_root)
                print(f"[cache] features_root → {feat_root}")
            if ac is _CACHE_MISS:
                frames = _load_allowed_frames(FILTERED_FRAMES_ROOT, vid, vp.stem)
                frames = _apply_custom_exclusions(frames, adir)
                Handler.allowed_frames = frames
                _save_allowed_frames_cache(adir, frames, ic)
                print(f"[cache] allowed_frames → {len(frames) if frames else 'none'}")
            if feat_root is not None:
                clips = compute_temporal_clips(
                    feat_root, vp.stem,
                    clip_length=CLIP_LENGTH,
                    top_fraction=TEMPORAL_TOP_FRACTION,
                    max_frame=TEMPORAL_MAX_FRAME,
                    cache_dir=adir,
                    allowed_frames=Handler.allowed_frames,
                )
                Handler.temporal_clips = clips
        Handler.allowed_frames = _startup_allowed          # set BEFORE bg thread reads it
        threading.Thread(target=_bg_startup, daemon=True).start()

    Handler.video = video
    Handler.tracker = tracker
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n{'='*58}")
    print(f"   http://localhost:{PORT}")
    print(f"   ssh -L {PORT}:localhost:{PORT} user@host")
    print(f"{'='*58}")
    if video is not None:
        print(f"\n   {video.total} samples @ {SAMPLE_FPS} fps — {_last_video.name}")
        print(f"   Autosave: {tracker.autosave_dir}")
    else:
        print(f"\n   Open http://localhost:{PORT} to pick a video from {VIDEO_PICKER_DIR}")
    print()

    try: server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving…")
        if video is not None:
            tracker._autosave()
        print("Done.")
        if video is not None:
            video.close()
        server.server_close()
