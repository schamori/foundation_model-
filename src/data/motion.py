"""
Offline motion map preprocessing for V-JEPA.

Computes optical flow magnitude maps per frame pair using RAFT (preferred)
or OpenCV Farneback (fallback). Output is used for motion-guided masking.

Usage:
    python -m src.data.motion --frames-root /path/to/frames --output /path/to/motion_maps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_raft():
    """Try to load RAFT from torchvision."""
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        return model.eval()
    except ImportError:
        return None


def _compute_flow_raft(
    model: torch.nn.Module,
    frame1: np.ndarray,
    frame2: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute optical flow magnitude using RAFT."""
    # Convert BGR/RGB (H, W, 3) to (1, 3, H, W) float tensor
    t1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    t2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        flows = model(t1, t2)
        flow = flows[-1]  # last iteration's flow: (1, 2, H, W)

    magnitude = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)  # (1, H, W)
    return magnitude.squeeze().cpu().numpy()


def _compute_flow_farneback(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute optical flow magnitude using OpenCV Farneback."""
    import cv2
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return magnitude


def compute_motion_maps(
    frames_root: Path,
    output_dir: Path,
    videos: list[str] | None = None,
    device: str = "cuda",
    max_size: int = 256,
) -> None:
    """Compute motion magnitude maps for all videos.

    Args:
        frames_root: directory with [category/]video_name/frame_NNNNNN.jpg
        output_dir: where to save motion maps as .npy files
        videos: if set, only process these video subdirectories
        device: torch device for RAFT
        max_size: resize frames to this max dimension for speed
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    raft_model = _load_raft()
    if raft_model is not None:
        raft_model = raft_model.to(device)
        print("[motion] Using RAFT for optical flow")
        use_raft = True
    else:
        print("[motion] RAFT unavailable, using OpenCV Farneback")
        use_raft = False

    # Discover videos
    from .dataset import discover_frames
    video_dict = discover_frames(frames_root)

    if videos:
        video_dict = {k: v for k, v in video_dict.items() if k in videos}

    print(f"[motion] Processing {len(video_dict)} videos → {output_dir}")

    for vkey, frame_paths in tqdm(video_dict.items(), desc="Motion maps"):
        vid_output = output_dir / vkey
        vid_output.mkdir(parents=True, exist_ok=True)

        prev_frame = None
        for i, fp in enumerate(frame_paths):
            from PIL import Image
            with Image.open(fp) as img:
                img = img.convert("RGB")
                # Resize for speed
                w, h = img.size
                scale = min(max_size / max(w, h), 1.0)
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)))
                frame = np.array(img)

            if prev_frame is not None:
                if use_raft:
                    mag = _compute_flow_raft(raft_model, prev_frame, frame, device)
                else:
                    mag = _compute_flow_farneback(prev_frame, frame)

                out_path = vid_output / f"{fp.stem}.npy"
                np.save(out_path, mag.astype(np.float16))

            prev_frame = frame

    print(f"[motion] Done. Saved to {output_dir}")


def load_motion_map(
    motion_dir: Path, video_name: str, frame_stem: str
) -> np.ndarray | None:
    """Load a precomputed motion map."""
    path = motion_dir / video_name / f"{frame_stem}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def motion_to_patch_bias(
    motion_map: np.ndarray, n_h: int, n_w: int
) -> torch.Tensor:
    """Convert pixel-level motion map to patch-level bias weights.

    Args:
        motion_map: (H, W) float32 motion magnitude
        n_h, n_w: number of patches in height/width

    Returns:
        (n_h, n_w) tensor of average motion per patch
    """
    t = torch.from_numpy(motion_map).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pooled = F.adaptive_avg_pool2d(t, (n_h, n_w))
    return pooled.squeeze()  # (n_h, n_w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute motion maps for V-JEPA")
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=256)
    args = parser.parse_args()

    compute_motion_maps(
        frames_root=args.frames_root,
        output_dir=args.output,
        videos=args.videos,
        device=args.device,
        max_size=args.max_size,
    )


if __name__ == "__main__":
    main()
