"""
Cross-video retrieval evaluation.

Extracts embeddings from eval_frames_root using the trained backbone,
then runs cross-video nearest-neighbour retrieval using phase labels
to measure phase accuracy and surgery-type accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from . import BaseEvaluator, register_evaluator

if True:  # TYPE_CHECKING
    from ..config import Config


# ---------------------------------------------------------------------------
# Helpers (self-contained to avoid heavy imports from datasplitting)
# ---------------------------------------------------------------------------

def _build_frame_phases(
    phases: list[dict], n_frames: int, all_codes: list[str],
) -> np.ndarray:
    """Map phase timestamps to per-frame phase indices at 1 fps."""
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    frame_phases = np.full(n_frames, -1, dtype=np.int32)
    for phase in phases:
        start = int(phase["start_ms"] // 1000)
        end = int(phase["end_ms"] // 1000)
        idx = code_to_idx.get(phase["code"], -1)
        if idx < 0:
            continue
        lo = max(0, start)
        hi = min(n_frames, end + 1)
        frame_phases[lo:hi] = idx
    return frame_phases


def _surgery_type(video_key: str) -> str:
    if "/" in video_key:
        return video_key.split("/", 1)[0]
    return video_key


def _cross_video_retrieval(
    video_embs: dict[str, np.ndarray],
    video_phases: dict[str, np.ndarray],
    device: str = "cuda",
    beta: float = 0.0,
) -> dict:
    """For each frame, find NN in all other videos. Returns accuracy metrics.

    If beta > 0, only count a retrieval if cosine similarity >= beta.
    """
    import faiss

    # Free GPU memory from training/embedding extraction before FAISS
    if "cuda" in device:
        torch.cuda.empty_cache()

    use_gpu = "cuda" in device and hasattr(faiss, "StandardGpuResources")
    gpu_res = None
    if use_gpu:
        gpu_res = faiss.StandardGpuResources()
        # Limit FAISS temp memory to 512 MB to avoid OOM
        gpu_res.setTempMemory(512 * 1024 * 1024)

    video_keys = sorted(video_embs.keys())
    vid_surgery = [_surgery_type(vk) for vk in video_keys]

    all_embs, all_vid_idx, all_phase = [], [], []
    for vi, vkey in enumerate(video_keys):
        emb = video_embs[vkey].astype(np.float32)
        phases = video_phases[vkey]
        n = min(len(emb), len(phases))
        for fi in range(n):
            if phases[fi] >= 0:
                all_embs.append(emb[fi])
                all_vid_idx.append(vi)
                all_phase.append(phases[fi])

    all_embs = np.vstack(all_embs).astype(np.float32)
    all_vid_idx = np.array(all_vid_idx, dtype=np.int64)
    all_phase = np.array(all_phase, dtype=np.int32)

    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embs = (all_embs / norms).astype(np.float32)

    d = all_embs.shape[1]
    phase_correct = stype_correct = total = 0
    per_phase_correct: dict[int, int] = {}
    per_phase_total: dict[int, int] = {}
    per_stype_correct: dict[str, int] = {}
    per_stype_total: dict[str, int] = {}

    for vi in range(len(video_keys)):
        q_mask = all_vid_idx == vi
        if not q_mask.any():
            continue

        q_embs = all_embs[q_mask]
        q_phases = all_phase[q_mask]

        other_mask = ~q_mask
        other_embs = all_embs[other_mask]
        other_phase = all_phase[other_mask]
        other_vid_idx = all_vid_idx[other_mask]

        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(other_embs)
        # Use CPU for large indices to avoid FAISS GPU OOM / CUBLAS assertion
        if use_gpu and other_embs.nbytes < 256 * 1024 * 1024:  # <256 MB
            other_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
            D, I = other_index.search(q_embs, 1)
        else:
            D, I = cpu_index.search(q_embs, 1)

        q_stype = vid_surgery[vi]
        for qi_local in range(len(q_embs)):
            nn = I[qi_local, 0]
            if nn < 0:
                continue
            if beta > 0 and D[qi_local, 0] < beta:
                continue
            nn_phase = other_phase[nn]
            nn_stype = vid_surgery[other_vid_idx[nn]]
            q_phase = q_phases[qi_local]

            p_match = int(nn_phase == q_phase)
            phase_correct += p_match
            per_phase_correct[q_phase] = per_phase_correct.get(q_phase, 0) + p_match
            per_phase_total[q_phase] = per_phase_total.get(q_phase, 0) + 1

            s_match = int(nn_stype == q_stype)
            stype_correct += s_match
            per_stype_correct[q_stype] = per_stype_correct.get(q_stype, 0) + s_match
            per_stype_total[q_stype] = per_stype_total.get(q_stype, 0) + 1

            total += 1

    skipped = int(np.sum(all_vid_idx >= 0)) - total  # frames skipped by beta filter
    return {
        "phase_accuracy": phase_correct / total if total > 0 else 0.0,
        "stype_accuracy": stype_correct / total if total > 0 else 0.0,
        "total": total,
        "skipped_by_beta": skipped,
        "per_phase_accuracy": {
            pi: per_phase_correct.get(pi, 0) / per_phase_total[pi]
            for pi in per_phase_total
        },
        "per_stype_accuracy": {
            st: per_stype_correct.get(st, 0) / per_stype_total[st]
            for st in per_stype_total
        },
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

class _FrameDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int = 224):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def _extract_embeddings(
    backbone: torch.nn.Module,
    frame_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
) -> np.ndarray:
    from ..model.backbone import pool_backbone_output

    ds = _FrameDataset(frame_paths, image_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    all_embs = []
    backbone = backbone.to(device).eval()
    with torch.no_grad():
        for batch, _ in loader:
            emb = pool_backbone_output(backbone, batch.to(device))
            all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs).astype(np.float32) if all_embs else np.empty((0, 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@register_evaluator
class CrossVideoRetrievalEvaluator(BaseEvaluator):
    """Cross-video phase retrieval evaluation on eval set."""

    name = "cross_video_retrieval"

    def evaluate_checkpoint(self, checkpoint_path: Path, epoch: int) -> dict:
        from ..model.backbone import load_backbone_from_checkpoint
        from ..data.read_phases import map_phases_to_videos, match_phases_to_frames

        eval_root = getattr(self.cfg, "eval_frames_root", None)
        if eval_root is None:
            result = {"epoch": epoch, "status": "skipped", "note": "eval_frames_root not set"}
            self.results.append(result)
            return result

        eval_root = Path(eval_root)
        if not eval_root.is_dir():
            result = {"epoch": epoch, "status": "skipped", "note": f"eval_frames_root not found: {eval_root}"}
            self.results.append(result)
            return result

        # Load phase labels and match to eval frame directories
        phase_data, found, missing = map_phases_to_videos()
        if found == 0:
            result = {"epoch": epoch, "status": "skipped", "note": "no phase labels found"}
            self.results.append(result)
            return result

        video_matches, _ = match_phases_to_frames(phase_data, eval_root)
        matched_keys = set(video_matches.values())

        if len(matched_keys) < 2:
            result = {"epoch": epoch, "status": "skipped",
                      "note": f"need >=2 phase-labeled videos in eval set, found {len(matched_keys)}"}
            self.results.append(result)
            return result

        all_codes = sorted({p["code"] for v in phase_data for p in v["phases"]})

        # Discover frames in eval_root
        from ..data.dataset import discover_frames
        all_videos = discover_frames(eval_root, self.cfg.exclude_folders)

        # Load backbone
        backbone = load_backbone_from_checkpoint(self.cfg, checkpoint_path)
        device = self.cfg.device

        # Extract embeddings and build phase arrays
        video_embs: dict[str, np.ndarray] = {}
        video_phases: dict[str, np.ndarray] = {}

        for vinfo in phase_data:
            if vinfo["name"] is None or vinfo["name"] not in video_matches:
                continue
            dir_key = video_matches[vinfo["name"]]
            if dir_key not in all_videos:
                continue

            frame_paths = all_videos[dir_key]
            if not frame_paths:
                continue

            emb = _extract_embeddings(
                backbone, frame_paths, device=device,
                batch_size=64, num_workers=4,
                image_size=getattr(self.cfg, "image_size", 224),
            )
            n_frames = len(emb)
            video_phases[dir_key] = _build_frame_phases(vinfo["phases"], n_frames, all_codes)
            video_embs[dir_key] = emb

        if len(video_embs) < 2:
            result = {"epoch": epoch, "status": "skipped",
                      "note": f"only {len(video_embs)} videos with embeddings, need >=2"}
            self.results.append(result)
            return result

        # Free backbone GPU memory before FAISS retrieval
        del backbone
        torch.cuda.empty_cache()

        beta = getattr(self.cfg, "retrieval_beta", 0.0)
        retrieval = _cross_video_retrieval(video_embs, video_phases, device=device, beta=beta)

        result = {
            "epoch": epoch,
            "phase_accuracy": retrieval["phase_accuracy"],
            "stype_accuracy": retrieval["stype_accuracy"],
            "total_frames": retrieval["total"],
            "n_videos": len(video_embs),
            "per_phase_accuracy": {
                all_codes[pi] if pi < len(all_codes) else f"phase_{pi}": acc
                for pi, acc in retrieval["per_phase_accuracy"].items()
            },
            "per_stype_accuracy": retrieval["per_stype_accuracy"],
        }
        self.results.append(result)
        return result
