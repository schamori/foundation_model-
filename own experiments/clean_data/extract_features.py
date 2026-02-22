"""
Extract ConvNext features from all images in a directory (recursively).
Saves features as .npy files in a mirrored folder structure.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextModel
from tqdm import tqdm
import json
from datetime import datetime


def load_model(model_name: str = "facebook/convnext-large-224-22k-1k"):
    """Load ConvNext model for feature extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ConvNextModel.from_pretrained(model_name).to(device)
    model.eval()

    processor = ConvNextImageProcessor.from_pretrained(
        model_name,
        size={"shortest_edge": 224},
        crop_size={"height": 224, "width": 224},
    )

    return model, processor, device


def get_all_image_paths(
        root_dir: Path,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
) -> list[Path]:
    """Recursively get all image paths from a directory."""
    paths = []
    for ext in extensions:
        paths.extend(root_dir.rglob(f"*{ext}"))
        paths.extend(root_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def extract_and_save_features(
        input_dir: Path,
        output_dir: Path,
        model,
        processor,
        device,
        batch_size: int = 16,
        skip_existing: bool = True,
):
    """
    Extract features from all images and save them as .npy files.
    Mirrors the input directory structure in the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Get all image paths
    print(f"Scanning {input_dir} for images...")
    all_image_paths = get_all_image_paths(input_dir)
    print(f"Found {len(all_image_paths)} images")

    if not all_image_paths:
        print("No images found!")
        return

    # Filter out already processed if skip_existing is True
    paths_to_process = []
    for img_path in all_image_paths:
        # Compute output path
        relative_path = img_path.relative_to(input_dir)
        feature_path = output_dir / relative_path.with_suffix(".npy")

        if skip_existing and feature_path.exists():
            continue
        paths_to_process.append((img_path, feature_path))

    print(
        f"Images to process: {len(paths_to_process)} (skipping {len(all_image_paths) - len(paths_to_process)} existing)")

    if not paths_to_process:
        print("All features already extracted!")
        return

    # Process in batches
    failed_images = []

    for i in tqdm(range(0, len(paths_to_process), batch_size), desc="Extracting features"):
        batch = paths_to_process[i:i + batch_size]

        images = []
        valid_items = []

        for img_path, feature_path in batch:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_items.append((img_path, feature_path))
            except Exception as e:
                print(f"\nError loading {img_path}: {e}")
                failed_images.append((img_path, str(e)))
                continue

        if not images:
            continue

        # Extract features
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values)
            batch_features = outputs.pooler_output.cpu().numpy()

        # Save each feature vector
        for j, (img_path, feature_path) in enumerate(valid_items):
            # Create output directory if needed
            feature_path.parent.mkdir(parents=True, exist_ok=True)

            # Save feature as .npy
            np.save(feature_path, batch_features[j])

    # Save metadata
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": "facebook/convnext-large-224-22k-1k",
        "feature_dim": 1536,  # ConvNext-Large pooled output dim
        "total_images": len(all_image_paths),
        "processed_images": len(paths_to_process) - len(failed_images),
        "failed_images": len(failed_images),
        "extraction_date": datetime.now().isoformat(),
    }

    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save failed images list if any
    if failed_images:
        failed_path = output_dir / "failed_images.json"
        with open(failed_path, "w") as f:
            json.dump([{"path": str(p), "error": e} for p, e in failed_images], f, indent=2)
        print(f"\n{len(failed_images)} images failed - see {failed_path}")

    print(f"\nFeatures saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    # ===== CONFIGURATION =====
    input_dir = Path(r"/media/HDD1/moritz/foundential/Extracted Frames/reference images")
    output_dir = Path(r"/media/HDD1/moritz/foundential/Extracted Frames Features/reference images")

    batch_size = 512
    skip_existing = True
    # =========================

    print("=" * 60)
    print("FEATURE EXTRACTION TOOL")
    print("=" * 60)

    # Load model
    print("\nLoading ConvNext model...")
    model, processor, device = load_model()

    # Extract and save features
    extract_and_save_features(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        processor=processor,
        device=device,
        batch_size=batch_size,
        skip_existing=skip_existing,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()