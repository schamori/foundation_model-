"""
DINOv2 frame feature extractor — standalone script.

Extracts DINOv2-base embeddings for all frames in a dataset directory
and saves them as .npy files (float16).

Usage:
    python dinov2_extract_features.py
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── Config ───────────────────────────────────────────────────────────────────
FRAMES_DIR = Path('/media/HDD1/moritz/Extracted Frames')
EMB_DIR = Path('/media/HDD1/moritz/Extracted Frames embeddings')
MODEL_NAME = 'facebook/dinov2-base'
BATCH_SIZE = 128
NUM_WORKERS = 8
FORCE_RECOMPUTE = False
GPU_ID = 0
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# ── Dataset / DataLoader ────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            with Image.open(rec['frame_path']) as img:
                pv = self.processor(images=img.convert('RGB'), return_tensors='pt')[
                    'pixel_values'].squeeze(0)
            return pv, rec['embedding_path'], True
        except Exception:
            return torch.zeros(3, 224, 224), rec['embedding_path'], False


def _collate(batch):
    pv, paths, ok = zip(*batch)
    return torch.stack(pv), list(paths), list(ok)


# ── Extraction ───────────────────────────────────────────────────────────────
def extract_embeddings(frames_dir: Path, emb_dir: Path, model, processor,
                       device, use_fp16: bool, force_recompute: bool = False):
    """Embed all frames in frames_dir/<video>/<frame.*> and save as .npy."""
    records = []
    for vdir in sorted(frames_dir.iterdir()):
        if not vdir.is_dir():
            continue
        vid = vdir.name
        for fp in sorted(vdir.iterdir()):
            if fp.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            records.append({
                'video_id': vid,
                'frame_path': str(fp),
                'embedding_path': str(emb_dir / vid / fp.with_suffix('.npy').name),
            })

    if not records:
        print('No frames found.')
        return

    total = len(records)
    if force_recompute:
        todo = records
    else:
        todo = [r for r in records if not Path(r['embedding_path']).exists()]

    print(f'Frames to embed: {len(todo):,} / {total:,}')
    if not todo:
        return

    loader = DataLoader(
        FrameDataset(todo, processor),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=use_fp16,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=_collate,
    )

    ac = (torch.autocast('cuda', dtype=torch.float16) if use_fp16
          else torch.autocast('cpu', enabled=False))

    saved = failed = 0
    with torch.no_grad(), ac:
        for pv, out_paths, valids in tqdm(loader, desc='Embedding'):
            pv = pv.to(device, non_blocking=True)
            if use_fp16:
                pv = pv.half()
            feats = model(pixel_values=pv).last_hidden_state.mean(dim=1)
            feats = feats.float().cpu().numpy().astype(np.float16)
            for feat, op, valid in zip(feats, out_paths, valids):
                if not valid:
                    failed += 1
                    continue
                p = Path(op)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, feat)
                saved += 1

    print(f'Saved {saved:,} embeddings  ({failed} failed)')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = device.type == 'cuda'

    print(f'Loading {MODEL_NAME}...')
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    if use_fp16:
        model = model.half()
    print(f'DINOv2 ready on {device}')

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Process each dataset directory under FRAMES_DIR
    dataset_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())
    print(f'Found {len(dataset_dirs)} dataset(s): {[d.name for d in dataset_dirs]}')

    for ddir in dataset_dirs:
        print(f'\n{"=" * 60}')
        print(f'Dataset: {ddir.name}')
        emb_subdir = EMB_DIR / ddir.name
        emb_subdir.mkdir(parents=True, exist_ok=True)
        extract_embeddings(ddir, emb_subdir, model, processor, device, use_fp16,
                           force_recompute=FORCE_RECOMPUTE)

    print('\nDone.')


if __name__ == '__main__':
    main()
