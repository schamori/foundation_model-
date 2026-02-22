"""
DINO Augmentations Test

Applies LemonFM's DINO augmentations to random images from the surgical dataset.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import random
import torch
import torchvision.transforms as T


class DINOAugmentations:
    """DINO augmentation pipeline for global and local crops."""

    def __init__(self):
        # ImageNet normalization stats
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Global crop augmentations (224x224)
        self.global_transforms = T.Compose([
            T.RandomResizedCrop(224, scale=(0.32, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
        ])

        # Global crop view 2 (with solarization)
        self.global_transforms_v2 = T.Compose([
            T.RandomResizedCrop(224, scale=(0.32, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            T.RandomSolarize(threshold=256, p=0.2),
        ])

        # Local crop augmentations (96x96)
        self.local_transforms = T.Compose([
            T.RandomResizedCrop(96, scale=(0.05, 0.32), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        ])

    def apply_global(self, image: Image.Image, view: int = 1) -> tuple[torch.Tensor, Image.Image]:
        """Apply global crop augmentation."""
        transform = self.global_transforms if view == 1 else self.global_transforms_v2
        aug_pil = transform(image)
        aug_tensor = self.normalize(T.ToTensor()(aug_pil))
        return aug_tensor, aug_pil

    def apply_local(self, image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
        """Apply local crop augmentation."""
        aug_pil = self.local_transforms(image)
        aug_tensor = self.normalize(T.ToTensor()(aug_pil))
        return aug_tensor, aug_pil


def get_random_image(images_root: Path) -> Path:
    """Get a random image from the dataset."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    all_images = []
    for ext in extensions:
        all_images.extend(images_root.rglob(f"*{ext}"))
        all_images.extend(images_root.rglob(f"*{ext.upper()}"))

    if not all_images:
        raise ValueError(f"No images found in {images_root}")

    return random.choice(all_images)


def visualize_augmentations(
    original: Image.Image,
    global_v1: Image.Image,
    global_v2: Image.Image,
    local_crops: list[Image.Image],
    save_path: str = "augmentations_test.png"
):
    """Visualize original image and all augmented versions."""
    n_local = len(local_crops)

    fig, axes = plt.subplots(2, 3 + n_local//2, figsize=(4 * (3 + n_local//2), 8))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Global crop view 1
    axes[1].imshow(global_v1)
    axes[1].set_title("Global Crop v1\n(224x224)", fontsize=10)
    axes[1].axis("off")

    # Global crop view 2 (with solarization)
    axes[2].imshow(global_v2)
    axes[2].set_title("Global Crop v2\n(224x224, +solarize)", fontsize=10)
    axes[2].axis("off")

    # Local crops
    for i, local_crop in enumerate(local_crops):
        axes[3 + i].imshow(local_crop)
        axes[3 + i].set_title(f"Local Crop {i+1}\n(96x96)", fontsize=10)
        axes[3 + i].axis("off")

    # Hide any unused subplots
    for i in range(3 + n_local, len(axes)):
        axes[i].axis("off")

    plt.suptitle("DINO Augmentations Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def main():
    # Configuration
    images_root = Path("/media/HDD1/moritz/foundential/Extracted Frames")
    n_local_crops = 6  # Number of local crops to generate

    print("=" * 60)
    print("DINO AUGMENTATIONS TEST")
    print("=" * 60)

    # Get random image
    image_path = get_random_image(images_root)
    print(f"\nLoading image: {image_path.relative_to(images_root)}")

    original = Image.open(image_path).convert("RGB")
    print(f"Original size: {original.size}")

    # Initialize augmentations
    augmenter = DINOAugmentations()

    # Apply augmentations
    print("\nApplying augmentations...")

    # Global crops
    _, global_v1 = augmenter.apply_global(original, view=1)
    _, global_v2 = augmenter.apply_global(original, view=2)

    # Local crops
    local_crops = []
    for i in range(n_local_crops):
        _, local_crop = augmenter.apply_local(original)
        local_crops.append(local_crop)

    print(f"✓ Generated 2 global crops (224x224)")
    print(f"✓ Generated {n_local_crops} local crops (96x96)")

    # Visualize
    visualize_augmentations(original, global_v1, global_v2, local_crops)


if __name__ == "__main__":
    main()
